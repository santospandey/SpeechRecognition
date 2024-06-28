import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class NepaliSoundDataset(Dataset):

    def __init__(
        self,
        annotations_file_path,
        audio_dir,
        transformation,
        target_sample_rate,
        device,
    ):
        self.annotaions = pd.read_csv(annotations_file_path, sep="\t")
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotaions)

    def _get_audio_sample_path(self, index):
        filename = self.annotaions.iloc[index, 0].strip()
        dir_name = filename[0]
        sub_dir_name = filename[:2]
        path = os.path.join(self.audio_dir, dir_name, sub_dir_name, filename + ".flac")
        return path

    def _get_audio_sample_output(self, index):
        return self.annotaions.iloc[index, 2]

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        output = self._get_audio_sample_output(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal = signal.to(self.device)
        # signal = self._resample(signal, sr)
        # signal = self._mix_down(signal)
        # signal = self.transformation(signal)
        return signal, output
