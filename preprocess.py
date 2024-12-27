import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch as t
import math
from utils import load_neg_mel_drom_disk

class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, filename, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        with open(filename, encoding='utf-8') as f:
            self.landmarks_frame = [line.strip().split('|') for line in f]
        self.root_dir = root_dir
        self.neg_mel_paths = 'train_val_mels'
    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def get_neg_mel(self, filename):
        audio_name = filename.strip().split('/')[4]
        audio_name = self.neg_mel_paths + '/' + audio_name + '.pt'
        text, neg_mel = load_neg_mel_drom_disk(audio_name)
        return text, neg_mel.T

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = self.landmarks_frame[idx][0]
        char_text = self.landmarks_frame[idx][1]

        text = np.asarray(text_to_sequence(char_text, [hp.cleaners]), dtype=np.int32)

        # load t2 wav without length limit
        # _, mel = self.get_neg_mel(wav_name)

        audio_name = wav_name.strip().split('/')[4]
        # load reference wav
        audio_name_ref = 'pos_train_val_mels' + '/' + audio_name + '.pt'
        # load t2 wav, frames equal to reference
        audio_name_t2_fixed_len = 't2_fixed_len' + '/' + audio_name + '.pt'
        ref_mel = t.load(audio_name_ref).T
        assert char_text == list(t.load(audio_name_t2_fixed_len).keys())[0]
        t2_mel = list(t.load(audio_name_t2_fixed_len).values())[0].T
        assert ref_mel.shape == t2_mel.shape

        # mel = np.load(wav_name[:-4] + '.pt.npy')
        # mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), ref_mel[:-1,:]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, ref_mel.shape[0] + 1)

        sample = {'text': text, 'ref_mel': ref_mel, 't2_fixed_len_mel': t2_mel, 'text_length': text_length, 'pos_mel': pos_mel, 'pos_text': pos_text}

        return sample
    
class PostDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav'
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mag = np.load(wav_name[:-4] + '.mag.npy')
        sample = {'mel':mel, 'mag':mag}

        return sample
    
def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):

        text = [d['text'] for d in batch]
        ref_mel = [d['ref_mel'] for d in batch]
        t2_mel = [d['t2_fixed_len_mel'] for d in batch]
        # mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        
        text = [i for i, _ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        ref_mel = [i for i, _ in sorted(zip(ref_mel, text_length), key=lambda x: x[1], reverse=True)]
        t2_mel = [i for i, _ in sorted(zip(t2_mel, text_length), key=lambda x: x[1], reverse=True)]
        # mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        ref_mel = _pad_mel(ref_mel)
        t2_mel = _pad_mel(t2_mel)
        # mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)


        return t.LongTensor(text), t.FloatTensor(ref_mel), t.FloatTensor(t2_mel), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
    
def collate_fn_postnet(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        
        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return t.FloatTensor(mel), t.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    if max_len % 2:
        max_len += 1
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset():
    return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def get_post_dataset():
    return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    if max_len % 2:
        max_len += 1
    return np.stack([_pad_one(x, max_len) for x in inputs])

