import torch
from torch import nn
from torch.nn.utils.rnn import *

class PolicyNetwork(nn.Module):
  # AdaMML policy network. Based on PolicyNet from https://github.com/IBM/AdaMML/blob/main/models/policy_net.py
  def __init__(self, modality_feat_dims, hidden_size=40, temperature=5.0):
    super().__init__()
    self.temperature = temperature
    embedded_dim = 256
    self.num_modality = len(modality_feat_dims)

    network_feature_dims = [min(feat_dim, hidden_size) for feat_dim in modality_feat_dims]
    self.feat_dim_reduction_fcs = nn.ModuleList([nn.Linear(ori_dim, target_dim) if ori_dim != target_dim else nn.Identity()
      for ori_dim, target_dim in zip(modality_feat_dims, network_feature_dims)])

    self.lstm = nn.LSTMCell(sum(network_feature_dims) + 2 * self.num_modality, embedded_dim)
    self.decision_fcs = nn.ModuleList([nn.Linear(embedded_dim, 2) for _ in range(self.num_modality)])

  def gumbel_softmax(self, logits):
    """
    :param logits: NxM, N is batch size, M is number of possible choices
    :return: Nx1: the selected index
    """
    distributions = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
    decisions = distributions[:, -1]
    return decisions

  def forward(self, xs, x_lens):

    # reduce input dimensions
    num_segments = xs[0].size(0)
    reduced = []

    for i in range(self.num_modality):
      reduced.append(self.feat_dim_reduction_fcs[i](xs[i]))
    reduced = torch.cat(reduced, dim=2)

    # pass through LSTM
    all_logits = []
    decisions = []
    for i in range(num_segments):
      if i == 0:
        lstm_in = torch.cat((reduced[i],
                              torch.zeros((reduced[i].shape[0], self.num_modality * 2),
                                          dtype=reduced[i].dtype, device=reduced[i].device)
                              ), dim=-1)
        h_x, c_x = self.lstm(lstm_in)  # h_x: Nxhidden, c_x: Nxhidden
      else:
        logits = logits.view((self.num_modality, -1, 2)).permute(1, 0, 2).contiguous().view(-1, 2 * self.num_modality)
        lstm_in = torch.cat((reduced[i], logits), dim=-1)

        h_x, c_x = self.lstm(lstm_in, (h_x, c_x))  # h_x: Nxhidden, c_x: Nxhidden

      logits = []
      for m_i in range(self.num_modality):
        tmp = self.decision_fcs[m_i](h_x)  # Nx2
        logits.append(tmp)
      logits = torch.cat(logits, dim=0)  # MNx2
      all_logits.append(logits.view(self.num_modality, -1, 2))
      selection = self.gumbel_softmax(logits)  # MNx1
      decisions.append(selection)
    decisions = torch.stack(decisions, dim=0).view(num_segments, self.num_modality, -1)
    all_logits = torch.stack(all_logits, dim=0)

    return decisions, all_logits

class ContextualLSTM(nn.Module):
  def __init__(self, modality_feat_dims, hidden_size=128, **kwargs):
    super().__init__()
    self.unimodal_lstms = nn.ModuleList([nn.LSTM(dim, hidden_size, **kwargs) for dim in modality_feat_dims])
    if kwargs.get("bidirectional", False):
      multiplier = 2
    else:
      multiplier = 1
    self.multimodal_lstm = nn.LSTM(hidden_size * len(modality_feat_dims) * multiplier, hidden_size, **kwargs)
    self.hidden_size = hidden_size * multiplier
    self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

  def forward(self, xs, x_lens, decisions):
    # unimodal
    contextual_embeddings = list()
    for i, X in enumerate(xs):
      packed_X = pack_padded_sequence(X * decisions[:,i].unsqueeze(dim=2), x_lens, enforce_sorted=False)
      packed_out, _ = self.unimodal_lstms[i](packed_X)
      X, _ = pad_packed_sequence(packed_out)
      contextual_embeddings.append(X) # modality decision is at dim 1

    # multimodal
    packed_X = pack_padded_sequence(
        self.dropout(torch.cat(contextual_embeddings, dim=2) * 3.1 / (decisions.sum(dim=1).unsqueeze(dim=2) + 0.1)),
        x_lens, enforce_sorted=False)
    packed_out, _ = self.multimodal_lstm(packed_X)
    X, lengths = pad_packed_sequence(packed_out)
    
    return X, lengths

class AdaMML(nn.Module):
  def __init__(self, policy, fusion, linear_layer_sizes=[], activation=nn.GELU, dropout=0.2):
    super().__init__()
    self.policy= policy
    self.fusion = fusion

    if len(linear_layer_sizes) > 0:
      linear_layer_sizes = [fusion.hidden_size] + linear_layer_sizes
      linear_layers = list()
      for in_size, out_size in zip(linear_layer_sizes, linear_layer_sizes[1:]):
        linear_layers.extend([
          nn.Dropout(dropout),
          nn.Linear(in_size, out_size),
          activation(),
        ])

      self.linear = nn.Sequential(*linear_layers[:-1])
    else:
      self.linear = None

    self.dropout = nn.Dropout(dropout)

  def forward(self, xs, x_lens):
    xs = [self.dropout(x) for x in xs]
    decisions, _ = self.policy(xs, x_lens)
    predictor_out, predictor_lengths = self.fusion(xs, x_lens, decisions)
    
    last_element_selector = (predictor_lengths - 1).unsqueeze(1).repeat(1, self.fusion.hidden_size).unsqueeze(0).to(predictor_out.device)
    last_elements = torch.gather(predictor_out, 0, last_element_selector).squeeze(0)

    if self.linear is not None:
      last_elements = self.linear(last_elements)
    
    return last_elements, decisions

from tqdm import tqdm

decision_loss_target = 2

def run_model(model, dataloader):
  loss_stats = {
      "MAE loss": 0.0,
      "decision loss": 0.0
  }

  for xs, x_lens, y, x2_mask in tqdm(dataloader, desc="Train" if model.training else "Eval "):
    if model.training:
      optimizer.zero_grad() # clear calculated gradients

    x = [d.to(device) for d in xs]
    y = y.to(device).unsqueeze(1)
    
    with torch.cuda.amp.autocast():
      output, batch_decision = model(x, x_lens)
      mae_loss = criterion(output, y)
      decision_loss = 0.0

      for i, length in enumerate(x_lens):
        decision = batch_decision[:length, :, i]
        decision_loss += torch.abs(decision.sum(dim=1) - decision_loss_target).mean()
      decision_loss /= len(x_lens)
        
    loss_stats["MAE loss"] += mae_loss.item()
    loss_stats["decision loss"] += decision_loss.item()


    if model.training:
      # backprop loss
      scaler.scale(mae_loss + decision_loss).backward()
      scaler.step(optimizer)
      scaler.update()

  for k, v in loss_stats.items():
    loss_stats[k] = v / len(dataloader)

  return loss_stats

model = AdaMML(
    policy=PolicyNetwork(modality_feat_dims, hidden_size=40, temperature=5.0),
    fusion=ContextualLSTM(modality_feat_dims, hidden_size=64, bidirectional=True, num_layers=2, dropout=0.2),
    linear_layer_sizes=[16, 1]
)