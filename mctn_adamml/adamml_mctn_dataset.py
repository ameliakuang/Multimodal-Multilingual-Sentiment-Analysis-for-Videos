import pickle
import pandas as pd
from torch.utils import data

original_labels = pd.concat([pd.read_csv("gdrive/MyDrive/moseas_french/batch_1_2/processed_moseas_french_batch_1_2_avglabel.csv"),
                             pd.read_csv("gdrive/MyDrive/moseas_french/batch_3/processed_moseas_french_batch_3_avglabel.csv"),
                             pd.read_csv("gdrive/MyDrive/11-777 MMML/processed_moseas_spanish_batch1_2_avglabel.csv")
                             ]).reset_index(drop=True)


with open('/content/gdrive/MyDrive/11-777 MMML/data/cz_video_based_data_splits.pickle', 'rb') as handle:
  # train_segments, _, _ = pickle.load(handle)
  train_segments, dev_segments, test_segments = pickle.load(handle)

# with open('/content/gdrive/MyDrive/11-777 MMML/data/amelia_test_split.pickle', 'rb') as handle:
#   test_segments = pickle.load(handle)

# with open('/content/gdrive/MyDrive/11-777 MMML/data/cz_video_based_data_splits_spanish.pickle', 'rb') as handle:
#   train_segments_spanish, dev_segments, test_segments = pickle.load(handle)
#   train_segments = train_segments.union(train_segments_spanish[:100])
#   # train_segments = set(train_segments_spanish)
#   train_segments_spanish = set(train_segments_spanish)
#   dev_segments = set(dev_segments)
#   test_segments = set(test_segments)

# french features

with open('/content/gdrive/MyDrive/11-777 MMML/data/french_mfa_aligned_librosa_acoustic_full.pickle', 'rb') as handle:
    aligned_librosa_ids, aligned_librosa_feats = pickle.load(handle)
    librosa_dict = dict()
    for id, feat in zip(aligned_librosa_ids, aligned_librosa_feats):
      librosa_dict[id] = feat[:, 40:]

with open('/content/gdrive/MyDrive/11-777 MMML/data/french_mfa_aligned_openface_shuhao_full.pickle', 'rb') as handle:
    aligned_openface_ids, aligned_openface_feats = pickle.load(handle)
    openface_dict = dict()
    for id, feat in zip(aligned_openface_ids, aligned_openface_feats):
      openface_dict[id] = feat[:, 678:713]

with open('/content/gdrive/MyDrive/11-777 MMML/data/french_mfa_aligned_word_embed.pickle', 'rb') as handle:
    aligned_word_embed_ids, aligned_word_embed_feats = pickle.load(handle)
    word_dict = dict()
    for id, feat in zip(aligned_word_embed_ids, aligned_word_embed_feats):
      word_dict[id] = feat

# spanish features
with open('/content/gdrive/MyDrive/11-777 MMML/data/spanish_mfa_aligned_librosa_acoustic_full.pickle', 'rb') as handle:
    aligned_librosa_ids, aligned_librosa_feats = pickle.load(handle)
    for id, feat in zip(aligned_librosa_ids, aligned_librosa_feats):
      librosa_dict[id] = feat[:, 40:]

with open('/content/gdrive/MyDrive/11-777 MMML/data/spanish_mfa_aligned_openface_shuhao_full.pickle', 'rb') as handle:
    aligned_openface_ids, aligned_openface_feats = pickle.load(handle)
    for id, feat in zip(aligned_openface_ids, aligned_openface_feats):
      openface_dict[id] = feat[:, 678:713]

with open('/content/gdrive/MyDrive/11-777 MMML/data/spanish_mfa_aligned_word_embed.pickle', 'rb') as handle:
    aligned_word_embed_ids, aligned_word_embed_feats = pickle.load(handle)
    for id, feat in zip(aligned_word_embed_ids, aligned_word_embed_feats):
      word_dict[id] = feat

train_data = list()
train_labels = list()
dev_data = list()
dev_labels = list()
test_data = list()
test_labels = list()

for _, (key, sentiment) in original_labels[["key", "sentiment"]].iterrows():
  if -0.5 <= sentiment <= 0.5:
    continue
  # if key in librosa_dict and key in word_dict:
  if key in word_dict and key in openface_dict:
    if key in train_segments:
      feats = train_data
      labels = train_labels
    elif key in dev_segments:
      feats = dev_data
      labels = dev_labels
    elif key in test_segments:
      feats = test_data
      labels = test_labels
    else:
      continue


    # feats.append((librosa_dict[key], word_dict[key], openface_dict.get(key))) # None if non-existent
    feats.append(( 
        librosa_dict[key],
        word_dict[key],
        openface_dict[key],
        ))
    labels.append(sentiment)

modality_feat_dims = [
                      next(iter(librosa_dict.values())).shape[1],
                      next(iter(word_dict.values())).shape[1],
                      next(iter(openface_dict.values())).shape[1],              
                      ]

print(modality_feat_dims)

class SequenceData(data.Dataset):
  def __init__(self, data, label = None):
    self.data = data
    self.label = np.array(label)
    self.length = len(self.data)

  def __len__(self):
    return self.length

  def __getitem__(self, i):
    data_tensors = list()

    for modal_idx, modal_feats in enumerate(self.data[i]):
      if modal_feats is not None:
        data_tensors.append(torch.from_numpy(modal_feats.astype(np.float32)))
      else:
        data_tensors.append(None)

    if self.label is None:
      return data_tensors
    else:
      # label = int(np.round(self.label[i] + 3))
      label = self.label[i]
      return data_tensors, torch.tensor(label)

  def collate_fn(batch):
    batch_X, batch_Y = zip(*batch) # unzip a list of tuples to lists
    batch_X = list(zip(*batch_X)) # transpose to list of batched data

    X_lens = torch.LongTensor([len(seq) for seq in batch_X[0]])

    padded_data = list()
    for batch_data in batch_X:
        padded_data.append(pad_sequence([data for data in batch_data if data is not None]))
    
    if len(batch_X) > 2:
      x2_mask = torch.tensor([data is not None for data in batch_X[2]]) # hard-coded
    else:
      x2_mask = None

    return padded_data, X_lens, torch.tensor(batch_Y), x2_mask
  
  def collate_test(batch_X):
    X_lens = torch.LongTensor([len(seq) for seq in batch_X])
    return pad_sequence(batch_X), X_lens
