import torch, pickle, pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MOSEICategorical(Dataset):    
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid
    
    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class MOSEIRegression(Dataset):
    # active if text / video / audio
    def __init__(self, path, train=True, active_modal=[True, True, True], mode="french"):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText1,\
            self.videoAudio, self.videoVisual, \
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'))
        
        if mode == "spanish-all-test":
            self.keys = self.trainVid + self.testVid
        else:
            self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        self.active_modal = active_modal

    def __getitem__(self, index):
        vid = self.keys[index]
        # import numpy as np
        # labels = np.array(self.videoLabels[vid])
        # print("dataloader.py", np.logical_and(labels < 0.5, labels>-0.5).sum())
        return self._deactive((torch.FloatTensor(self.videoText1[vid]),\
                              torch.FloatTensor(self.videoVisual[vid]),\
                              torch.FloatTensor(self.videoAudio[vid]),\
                              torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                                  self.videoSpeakers[vid]]),\
                              torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                              torch.FloatTensor(self.videoLabels[vid])))

    def _deactive(self, data):
        active = self.active_modal
        for i, act in enumerate(active):
            if not act:
                data[i] = torch.zeros(data[i].shape)
        return data

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i].tolist()) if i<5 else pad_sequence(dat[i].tolist(), True) for i in dat]

class MOSEIRegression_NCE(Dataset):
    # active if text / video / audio
    def __init__(self, path, train=True, active_modal=[True, True, True], mode="french"):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText1,\
            self.videoText2, self.videoAudio, self.videoVisual, \
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'))
        
        if mode == "spanish-all-test":
            self.keys = self.trainVid + self.testVid
        else:
            self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        self.active_modal = active_modal

    def __getitem__(self, index):
        vid = self.keys[index]
        # import numpy as np
        # labels = np.array(self.videoLabels[vid])
        # print("dataloader.py", np.logical_and(labels < 0.5, labels>-0.5).sum())
        return self._deactive((torch.FloatTensor(self.videoText1[vid]),\
                              torch.FloatTensor(self.videoText2[vid]),\
                              torch.FloatTensor(self.videoVisual[vid]),\
                              torch.FloatTensor(self.videoAudio[vid]),\
                              torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                                  self.videoSpeakers[vid]]),\
                              torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                              torch.FloatTensor(self.videoLabels[vid])))

    def _deactive(self, data):
        active = self.active_modal
        for i, act in enumerate(active):
            if not act:
                data[i] = torch.zeros(data[i].shape)
        return data

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i].tolist()) if i<5 else pad_sequence(dat[i].tolist(), True) for i in dat]



class MOSEIRegressionNCEBi(Dataset):
    # active if text / video / audio
    def __init__(self, path1, path2, train=True, active_modal=[True, True, True], mode="french"):
        videoIDs1, videoSpeakers1, videoLabels1, videoText11,\
            videoAudio1, videoVisual1, \
            trainVid1, testVid1 = pickle.load(open(path1, 'rb'))
        videoIDs2, videoSpeakers2, videoLabels2, videoText12,\
            videoText22, videoAudio2, videoVisual2, \
            trainVid2, testVid2 = pickle.load(open(path2, 'rb'))
        
        videoIDs1.update(videoIDs2)
        self.videoIDs = videoIDs1
        videoSpeakers1.update(videoSpeakers2)
        self.videoSpeakers = videoSpeakers1
        videoLabels1.update(videoLabels2)
        self.videoLabels = videoLabels1
        videoText11.update(videoText12)
        self.videoText1 = videoText11
        # videoText21.update(videoText22)
        # self.videoText2 = videoText21
        videoAudio1.update(videoAudio2)
        self.videoAudio = videoAudio1
        videoVisual1.update(videoVisual2)
        self.videoVisual = videoVisual1
        trainVid1 += (trainVid2)
        self.trainVid = trainVid1
        testVid1 += (testVid2)
        self.testVid = testVid1
        
        if mode == "spanish-all-test":
            self.keys = self.trainVid + self.testVid
        else:
            self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        self.active_modal = active_modal

    def __getitem__(self, index):
        vid = self.keys[index]
        # import numpy as np
        # labels = np.array(self.videoLabels[vid])
        # print("dataloader.py", np.logical_and(labels < 0.5, labels>-0.5).sum())
        return self._deactive((torch.FloatTensor(self.videoText1[vid]),\
                              # torch.FloatTensor(self.videoText2[vid]),\
                              torch.FloatTensor(self.videoVisual[vid]),\
                              torch.FloatTensor(self.videoAudio[vid]),\
                              torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                                  self.videoSpeakers[vid]]),\
                              torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                              torch.FloatTensor(self.videoLabels[vid])))

    def _deactive(self, data):
        active = self.active_modal
        for i, act in enumerate(active):
            if not act:
                data[i] = torch.zeros(data[i].shape)
        return data

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i].tolist()) if i<5 else pad_sequence(dat[i].tolist(), True) for i in dat]


class MOSEIRegressionBi(Dataset):
    # active if text / video / audio
    def __init__(self, path1, path2, train=True, active_modal=[True, True, True], mode="french"):
        videoIDs1, videoSpeakers1, videoLabels1, videoText11,\
            videoAudio1, videoVisual1, \
            trainVid1, testVid1 = pickle.load(open(path1, 'rb'))
        videoIDs2, videoSpeakers2, videoLabels2, videoText12,\
            videoText22, videoAudio2, videoVisual2, \
            trainVid2, testVid2 = pickle.load(open(path2, 'rb'))
        
        videoIDs1.update(videoIDs2)
        self.videoIDs = videoIDs1
        videoSpeakers1.update(videoSpeakers2)
        self.videoSpeakers = videoSpeakers1
        videoLabels1.update(videoLabels2)
        self.videoLabels = videoLabels1
        videoText11.update(videoText12)
        self.videoText1 = videoText11
        # videoText21.update(videoText22)
        # self.videoText2 = videoText21
        videoAudio1.update(videoAudio2)
        self.videoAudio = videoAudio1
        videoVisual1.update(videoVisual2)
        self.videoVisual = videoVisual1
        trainVid1 += (trainVid2)
        self.trainVid = trainVid1
        testVid1 += (testVid2)
        self.testVid = testVid1
        
        if mode == "spanish-all-test":
            self.keys = self.trainVid + self.testVid
        else:
            self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        self.active_modal = active_modal

    def __getitem__(self, index):
        vid = self.keys[index]
        # import numpy as np
        # labels = np.array(self.videoLabels[vid])
        # print("dataloader.py", np.logical_and(labels < 0.5, labels>-0.5).sum())
        return self._deactive((torch.FloatTensor(self.videoText1[vid]),\
                              # torch.FloatTensor(self.videoText2[vid]),\
                              torch.FloatTensor(self.videoVisual[vid]),\
                              torch.FloatTensor(self.videoAudio[vid]),\
                              torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                                  self.videoSpeakers[vid]]),\
                              torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                              torch.FloatTensor(self.videoLabels[vid])))

    def _deactive(self, data):
        active = self.active_modal
        for i, act in enumerate(active):
            if not act:
                data[i] = torch.zeros(data[i].shape)
        return data

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i].tolist()) if i<5 else pad_sequence(dat[i].tolist(), True) for i in dat]
