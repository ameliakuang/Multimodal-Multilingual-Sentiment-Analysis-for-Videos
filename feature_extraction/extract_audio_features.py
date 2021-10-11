from google.colab import drive

drive.mount('/content/gdrive')

!cd gdrive/MyDrive/11777_data

!wget http://immortal.multicomp.cs.cmu.edu/cache/multilingual/french/zips/french_amt_b1_b2.zip

!mkdir moseas

!unzip -q gdrive/MyDrive/11777_data/french_amt_b1_b2.zip -d moseas

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# export BASE_DIR=moseas/zipper_french/Video
# 
# for video_name in $(ls $BASE_DIR); do
#   mkdir -p extracted_audio/$video_name
#   for video_segment in $(cd $BASE_DIR/$video_name && ls *.mp4); do
#     ffmpeg -i $BASE_DIR/$video_name/$video_segment -vn -acodec copy extracted_audio/$video_name/$video_segment.aac;
#   done;
# done;

!rm -r extracted_audio
!mkdir extracted_audio

!tar -xf gdrive/MyDrive/11777_data/moseas_audios.tgz -C .

import librosa
import numpy as np
import os, glob

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    mel = librosa.feature.melspectrogram(X, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
    for f in [mfccs, mel, chroma, contrast, tonnetz]:
      print(f.shape)

    return np.concatenate([mfccs, mel, chroma, contrast, tonnetz])

import warnings
from tqdm import tqdm
warnings.simplefilter('ignore')

from multiprocessing import Pool

def extract_wrapper(params):
  parent_path, segment = params
  try:
    features = extract_feature(parent_path + "/" + segment)
  except:
    features = None
  key_end_idx = 12 + segment[12:].find("_")
  return segment[:key_end_idx], features

def parse_audio_files(parent_dir):
    process_list = []
    audio_features = []
    segment_ids = []

    for video_name in os.listdir(parent_dir):
        for segment in os.listdir(parent_dir + "/" + video_name):
            process_list.append((parent_dir + "/" + video_name, segment))

    with Pool(processes=4) as pool:
      for id, feat in tqdm(pool.imap(extract_wrapper, process_list), desc="process", total=len(process_list)):
        if feat is not None:
          audio_features.append(feat)
          segment_ids.append(id)
      
    return segment_ids, audio_features

ids, feat = parse_audio_files("extracted_audio")

import pickle

with open('features_with_id.pickle', 'wb') as handle:
    pickle.dump((ids, feat), handle, protocol=pickle.HIGHEST_PROTOCOL)