import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model import AudioCNN  # custom CNN model for audio classification


# normalize spectrograms
class NormalizeSpec(nn.Module):
    def forward(self, spec):
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + 1e-6)


# custom dataset wrapper for ESC-50
class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None, classes=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # folds 1-4 for train, 5 for validation
        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        # class names → integer indices
        if classes is None:
            self.classes = sorted(self.metadata['category'].unique())
        else:
            self.classes = classes
                
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        # load audio
        row = self.metadata.iloc[index]
        audio_path = Path(self.data_dir) / "audio" / row['filename']
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # stereo → mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # apply preprocessing (MelSpectrogram etc.)
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, torch.tensor(row['label'], dtype=torch.long)


# mixup augmentation
def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1-lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# mixup loss wrapper
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)


def train():
    from datetime import datetime

    esc50_dir = Path("./ESC-50-master")
    metadata_file = esc50_dir/"meta"/"esc50.csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'./tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    # training transforms: resample → mel → augment
    train_transform = nn.Sequential(
        T.Resample(orig_freq=44100, new_freq=22050),
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        NormalizeSpec(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    # validation transforms: resample → mel (no augment)
    val_transform = nn.Sequential(
        T.Resample(orig_freq=44100, new_freq=22050),
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        NormalizeSpec(),
    )

    print("Training... ESC-50 dataset is available at", esc50_dir)
    meta = pd.read_csv(metadata_file)
    classes = sorted(meta['category'].unique())

    # datasets
    train_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=metadata_file,
        split="train",
        transform=train_transform,
        classes=classes
    )
    val_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=metadata_file,
        split="val",
        transform=val_transform,
        classes=classes
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model + device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    # optimizer, scheduler, and loss
    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )

    best_accuracy = 0.0
    best_val_loss = float('inf')
    patience = 20
    trigger_times = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # training loop
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100, ascii=True, file=sys.stdout, leave=True)
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            # randomly apply mixup augmentation ~40% of the time
            if np.random.random() > 0.6:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # validation loop
        model.eval()
        correct, total, val_loss = 0, 0, 0

        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)

                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(val_dataloader)

            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', accuracy, epoch)

            print(f'Epoch {epoch+1} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%')

            # save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_accuracy = accuracy
                trigger_times = 0

                os.makedirs('./saved_models', exist_ok=True)
                best_model_path = './saved_models/best_model.pth'

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch,
                    'classes': train_dataset.classes,
                    'sample_rate': 22050,
                    'mel_params': {
                        'n_fft': 1024,
                        'hop_length': 256,
                        'n_mels': 128,
                        'f_min': 0,
                        'f_max': 11025
                    }
                }, best_model_path)
                print(f'New best model saved with Val Loss: {avg_val_loss:.4f}')
            else:
                trigger_times += 1
                print(f'Validation loss did not improve. Trigger: {trigger_times}/{patience}')
                if trigger_times >= patience:
                    print(f'Early stopping at epoch {epoch+1}!')
                    break

        if trigger_times < patience:
            print("Training finished naturally without early stopping.")

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%' )


if __name__ == "__main__":
    train()
