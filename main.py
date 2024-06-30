import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchaudio
from comet_ml import Experiment
from torch.utils.data import random_split

from custom_nepali_dataset import NepaliSoundDataset
from model import SpeechRecognitionModel
from text_transform import TextTransform, cer, wer

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def check_data_for_nan_inf(data):
    if torch.isnan(data).any() or torch.isinf(data).any():
        return None
    return True


def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    iter_meter,
    experiment,
):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data

            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            # print("....................... Evaluations ..............................")
            # print(f"Oputput {output}")
            # print(f"Label {labels}")
            # print(f"Input length {input_lengths}")
            # print(f"Label Lengths {label_lengths}")
            # print("...................................................................")

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            experiment.log_metric("loss", loss.item(), step=iter_meter.get())
            experiment.log_metric(
                "learning_rate", scheduler.get_lr(), step=iter_meter.get()
            )

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 10 == 0 or batch_idx == data_len:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(spectrograms),
                        data_len,
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


def GreedyDecoder(
    output, labels, label_lengths, blank_label=71, collapse_repeated=True
):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(
            text_transform.int_to_text(labels[i][: label_lengths[i]].tolist())
        )
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with experiment.test():
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = GreedyDecoder(
                    output.transpose(0, 1), labels, label_lengths
                )
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    experiment.log_metric("test_loss", test_loss, step=iter_meter.get())
    experiment.log_metric("cer", avg_cer, step=iter_meter.get())
    experiment.log_metric("wer", avg_wer, step=iter_meter.get())

    print(
        "Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            test_loss, avg_cer, avg_wer
        )
    )


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, _, utterance, _ in data:
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


def main(
    dataset,
    learning_rate,
    batch_size,
    epochs,
    experiment=Experiment(api_key="dummy_key", disabled=True),
):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 77,
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
    print(f"Train Loader: {train_loader}")

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        collate_fn=lambda x: data_processing(x, "valid"),
        **kwargs,
    )
    print(f"Test Loader {test_loader}")

    # # Iterate over the DataLoader
    # for batch, _data in enumerate(train_loader):
    #     # print(f"Batch: {batch}\n")
    #     (padded_tensors, sample_rates, transcripts, speaker_ids) = _data
    #     # Use the batch for training
    #     # print("Batch of padded tensors:", padded_tensors.shape)
    #     # Training code goes here

    model = SpeechRecognitionModel(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_class"],
        hparams["n_feats"],
        hparams["stride"],
        hparams["dropout"],
    ).to(device)

    print(f"Model: {model}")
    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    optimizer = optim.AdamW(model.parameters(), hparams["learning_rate"])
    criterion = nn.CTCLoss(blank=71).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=hparams["epochs"],
        anneal_strategy="linear",
    )

    # print(f"Optimizer {optimizer}")
    # print(f"Criterion {criterion}")
    # print(f"Scheduler {scheduler}")

    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
            experiment,
        )
        test(model, device, test_loader, criterion, epoch, iter_meter, experiment)

    # Save the model and optimizer state dicts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "speech_recognition.pth",
    )
    print("Trained model successfully")


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
    # signal, sample_rate, output, speaker_id = dataset[400]
    # print(
    #     f"Signal {signal}\n Output {output} \nSample rate {sample_rate}\nSpeaker Id {speaker_id}"
    # )
    # print(f"Shape {signal.shape}")

    # Setting Comet Experiment
    comet_api_key = "uAoybvu8H90J4enDCx6FHdxKO"  # add your api key here
    project_name = "speechrecognition"
    experiment_name = "speechrecognition-colab"

    if comet_api_key:
        experiment = Experiment(
            api_key=comet_api_key, project_name=project_name, parse_args=False
        )
        experiment.set_name(experiment_name)
        experiment.display()
    else:
        experiment = Experiment(api_key="dummy_key", disabled=True)

    learning_rate = 5e-4
    batch_size = 10
    epochs = 10
    main(dataset, learning_rate, batch_size, epochs)
