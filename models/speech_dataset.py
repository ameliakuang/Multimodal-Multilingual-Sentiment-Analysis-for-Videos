class SpeechData(data.Dataset):
  def __init__(self, data, label = None, apply_mask = False):
    self.data = data
    self.label = label
    self.length = len(self.data)
    self.masks = nn.Sequential(
      AddGaussianNoise(),
      torchaudio.transforms.FrequencyMasking(freq_mask_param=6),
      torchaudio.transforms.TimeMasking(time_mask_param=40)
    ) if apply_mask else None

  def __len__(self):
    return self.length

  def __getitem__(self, i):
    melspectrogram = torch.from_numpy(self.data[i].astype(np.float32)[168:, :])

    if self.label is None:
      return melspectrogram
    else:
      if self.masks is not None:
        melspectrogram = self.masks(melspectrogram.transpose(0, 1)).transpose(0, 1)
      
      # label = int(np.round(self.label[i] + 3))
      label = self.label[i]
      return melspectrogram, torch.tensor(label)