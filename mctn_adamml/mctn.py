import torch
from torch import nn
from torch.nn.utils.rnn import *

class RNNWithLinear(nn.Module):
  def __init__(self, rnn, layer_sizes=[], dropout_p = 0.2, activation=nn.GELU, is_vanilla_rnn=False):
    super(RNNWithLinear, self).__init__()
    
    self.rnn = rnn
    self.is_vanilla_rnn = is_vanilla_rnn
    
    if len(layer_sizes) > 0:
      linear_layers = list()
      for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        linear_layers.extend([
          nn.Dropout(dropout_p),
          nn.Linear(in_size, out_size),
          activation(),
        ])

      self.linear = nn.Sequential(*linear_layers[:-1])
    else:
      self.linear = None

  def forward(self, X, lengths):
    if self.is_vanilla_rnn:
      packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
      packed_out, (hidden_states, cell_states) = self.rnn(packed_X)
      X, lengths = pad_packed_sequence(packed_out)
    else:
      X, hidden_states, lengths = self.rnn(X, lengths)
    
    if self.linear is not None:
      X = self.linear(X)
    
    return X, hidden_states, lengths

class Encoder(nn.Module):
  def __init__(self, rnn, rnn_out_size, kqv_size):
    super(Encoder, self).__init__()

    self.rnn = rnn
    self.rnn_out_size = rnn_out_size
    self.key_transform = nn.Linear(rnn_out_size, kqv_size)
    self.value_transform = nn.Linear(rnn_out_size, kqv_size)
    
  def forward(self, x, lengths):
    # all 3-d output shape: L, N, C_out
    output, hidden, lengths = self.rnn(x, lengths)
    keys = self.key_transform(output)
    values = self.value_transform(output)
    return keys, values, lengths, output

attention_score_accumulator = list()
attention_energy_accumulator = list()

def attention(queries, keys, values, valid_mask):
  """
  @param queries: (Batch, kqv_size)
  @param keys, values: (Batch, Length, kqv_size)
  @param mask: (Batch)
  """
  energy = torch.bmm(keys, queries.unsqueeze(2)) # Batch, Length, 1 
  
  # based on http://juditacs.github.io/2018/12/27/masked-attention.html
  energy[~valid_mask.unsqueeze(2)] = float("-inf")
  
  attention_score = energy.softmax(dim=1) # Batch, Length, 1
  
  if DEBUG:
    attention_score_accumulator.append(attention_score.detach().cpu())
    attention_energy_accumulator.append(energy.detach().cpu())
  
  context = (attention_score * values).sum(dim=1)

  return context

class Decoder(nn.Module):
  def __init__(self, output_size, decoder_hidden_dim, key_value_size=128):
    super(Decoder, self).__init__()

    self.lstm_cells = nn.Sequential(
      nn.LSTMCell(input_size=2 * key_value_size, hidden_size=decoder_hidden_dim),
      nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)
    )

    self.linear = nn.Linear(2 * key_value_size, output_size)

    self.key_value_size = key_value_size

  def forward(self, key, value, encoder_len):
    '''
    Args:
        key :(Length, Batch, key_value_size) - Output of the Encoder Key projection layer
        value: (Length, Batch, key_value_size) - Output of the Encoder Value projection layer

    Return:
        predictions: the character perdiction probability 
    '''

    key = key.permute(1, 0, 2)
    value = value.permute(1, 0, 2)

    batch_size, key_seq_max_len, key_value_size = key.shape

    max_len = encoder_len.max().item()

    # Create the attention mask here (outside the for loop rather than inside) to aviod repetition
    idx = torch.arange(key_seq_max_len).repeat(batch_size, 1)
    attention_valid_mask = idx < encoder_len.unsqueeze(1)

    predictions = []
    prediction = torch.zeros(batch_size, 1).to(device)
    hidden_states = [None] * len(self.lstm_cells)

    # Initialize the context
    context = attention(torch.zeros(batch_size, self.key_value_size).to(device),
                        key, value, attention_valid_mask)
    
    output = torch.zeros(batch_size, self.key_value_size).to(device)

    output_context = torch.cat([output, context], dim=1)

    for i in range(max_len):
      output = output_context
      
      # loop through LSTM layers
      for i, cell in enumerate(self.lstm_cells):        
        hidden_states[i] = cell(output, hidden_states[i])
        output = hidden_states[i][0]

      # Compute attention from the output of the second LSTM Cell
      context = attention(output, key, value, attention_valid_mask)

      output_context = torch.cat([output, context], dim=1)
      prediction = self.linear(output_context)
      predictions.append(prediction)
    return torch.stack(predictions, dim=0)

class MCTNBimodal(nn.Module):
    def __init__(self, encoder, decoder, predictor=None, predictor_out_size=None, input_size=None):
        super(MCTNBimodal,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if predictor is not None:
          self.is_predicting = True
          self.predictor = predictor
          self.predictor_out_size = predictor_out_size
          self.linear = nn.Linear(predictor_out_size, 1)
        else:
          self.is_predicting = False
          self.linear = None
          if self.encoder.rnn_out_size != input_size:
            self.linear = nn.Linear(self.encoder.rnn_out_size, input_size)

    def forward(self, x, x_len):
        keys, values, lengths, output = self.encoder(x, x_len)
        decoded = self.decoder(keys, values, lengths)

        keys, values, lengths, _ = self.encoder(decoded, lengths)
        cycle_decoded = self.decoder(keys, values, lengths)

        if self.is_predicting:
          predictor_out, _, predictor_lengths = self.predictor(output, lengths)
          last_element_selector = (predictor_lengths - 1).unsqueeze(1).repeat(1, self.predictor_out_size).unsqueeze(0).to(predictor_out.device)
          
          # set all irrelevant values to 0
          idx = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, x.shape[1])
          predictions_invalid_mask = (idx >= x_len.unsqueeze(0)).to(decoded.device)
          decoded[predictions_invalid_mask] = 0.0
          cycle_decoded[predictions_invalid_mask] = 0.0
          last_elements = torch.gather(predictor_out, 0, last_element_selector).squeeze(0)
          output = self.linear(last_elements)
        elif self.linear is not None:
          output = self.linear(output)
          last_elements = None

        return decoded, cycle_decoded, output, last_elements

from tqdm import tqdm

def run_model(model, dataloader):
  cycle_loss_func = nn.MSELoss()
  loss_stats = {
      "MAE loss": 0.0,
      "decoded loss": 0.0,
      "cycle loss": 0.0
  }

  for xs, x_lens, y, x2_mask in tqdm(dataloader, desc="Train" if model.training else "Eval "):
    if model.training:
      optimizer.zero_grad() # clear calculated gradients

    x = [d.to(device) for d in xs]
    y = y.to(device).unsqueeze(1)
    
    with torch.cuda.amp.autocast():
      decoded, cycle_decoded, output = model[0](x[0], x_lens)
      
      decoded_loss = cycle_loss_func(x[1], decoded)
      cycle_loss = cycle_loss_func(x[0], cycle_decoded)

      if len(x[2]) > 0:
        decoded, cycle_decoded, output_new = model[1](x[0], x_lens)

        decoded_loss += cycle_loss_func(x[2], decoded[:x[2].size(0), x2_mask])
        cycle_loss += cycle_loss_func(x[0][:, x2_mask], cycle_decoded[:, x2_mask])
        output = output_new

      objective_loss = criterion(output, y)
      loss = objective_loss + 0.3 * decoded_loss + 0.3 * cycle_loss

    # accumuate stats
    loss_stats["MAE loss"] += objective_loss.item()
    loss_stats["decoded loss"] += decoded_loss.item()
    loss_stats["cycle loss"] += cycle_loss.item()

    if model.training:
      # backprop loss
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

  for k, v in loss_stats.items():
    loss_stats[k] = v / len(dataloader)

  return loss_stats

# model definition
key_query_value_size = 128
encoder_out_size = 512
in_out_size = sum(modality_feat_dims)

encoder = Encoder(
                  rnn=RNNWithLinear(nn.LSTM(in_out_size, 256, bidirectional=True, num_layers = 2, dropout = 0.2),
                                    is_vanilla_rnn = True),
                  rnn_out_size = encoder_out_size,
                  kqv_size = key_query_value_size
                )

# MCTN trimodal with shared encoder
model = nn.Sequential(
  MCTNBimodal(
      encoder,
      Decoder(
          output_size = in_out_size,
          decoder_hidden_dim = 512,
          key_value_size = key_query_value_size,
      ),
      input_size = in_out_size
  ),
  MCTNBimodal(
    encoder,
    Decoder(
        output_size = in_out_size,
        decoder_hidden_dim = 512,
        key_value_size = key_query_value_size,
    ),
    RNNWithLinear(nn.LSTM(encoder_out_size, 128, bidirectional=True, num_layers = 2, dropout = 0.2),
                        is_vanilla_rnn = True),
    predictor_out_size = 256,
    input_size = in_out_size
  )
)
