import os

import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class NepaliSoundDataset(Dataset):

    def __init__(self, annotations_file_path, audio_dir):
        self.annotaions = pd.read_csv(annotations_file_path, sep="\t")
        self.audio_dir = audio_dir

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

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        output = self._get_audio_sample_output(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, output


if __name__ == "__main__":
    annotations_file_path = (
        "/Users/santoshpandey/Desktop/ASR/code/data/OpenSLR/utt_spk_text.tsv"
    )
    audio_dir = "/Users/santoshpandey/Desktop/ASR/code/data/OpenSLR"
    dataset = NepaliSoundDataset(annotations_file_path, audio_dir)
    print(f"There are {len(dataset)} items")
    signal, output = dataset[1000]
    print(f"signal shape {signal.shape}")
    print(f"signal {signal}\n output {output}")
