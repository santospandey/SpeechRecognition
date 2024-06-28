import torch
import torch.nn as nn
import torch.utils.data as data
import torchaudio
from torch.utils.data import random_split

from custom_nepali_dataset import NepaliSoundDataset
from model import SpeechRecognitionModel
from text_transform import TextTransform

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for waveform, _, utterance, _, _, _ in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == "valid":
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception("data_type should be train or valid")
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = (
        nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        .unsqueeze(1)
        .transpose(2, 3)
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def main(dataset, learning_rate: float, batch_size: int, epochs: int):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Total dataset length
    dataset_length = len(dataset)

    # Define the split lengths
    train_length = int(0.8 * dataset_length)
    test_length = dataset_length - train_length

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Testing dataset length: {len(test_dataset)}")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=lambda x: data_processing(x, "train"),
        **kwargs,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        collate_fn=lambda x: data_processing(x, "valid"),
        **kwargs,
    )

    print(f"Train Loader {train_loader}")
    print(f"Test Loader {test_loader}")

    model = SpeechRecognitionModel(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_class"],
        hparams["n_feats"],
        hparams["stride"],
        hparams["dropout"],
    ).to(device)

    print(model)
    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )


if __name__ == "__main__":
    ANNOTATIONS_FILE_PATH = (
        "/Users/santoshpandey/Desktop/ASR/code/data/OpenSLR/utt_spk_text.tsv"
    )
    AUDIO_DIR = "/Users/santoshpandey/Desktop/ASR/code/data/OpenSLR"
    SAMPLE_RATE = 16000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    dataset = NepaliSoundDataset(
        ANNOTATIONS_FILE_PATH, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, device
    )
    print(f"There are {len(dataset)} items")
    signal, output = dataset[67]
    print(f"signal shape {signal.shape}")
    print(f"signal {signal}\n output {output}")

    learning_rate = 5e-4
    batch_size = 10
    epochs = 10
    main(dataset, learning_rate, batch_size, epochs)
