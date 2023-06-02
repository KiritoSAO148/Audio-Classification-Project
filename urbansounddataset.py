from torch.utils.data import Dataset, random_split
import pandas as pd
import torchaudio
import torch
from audio_preprocessing import AudioUtil
import os
import matplotlib.pyplot as plt

class UrbanSoundDataset(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id

def img(train_dl):
    image_folder = 'image'
    os.makedirs(image_folder, exist_ok=True)
    for i, (sgram, class_id) in enumerate(train_dl):
        for j in range(sgram.size(0)):
            for k in range(sgram.size(1)):
                aug_sgram = sgram[j, k].squeeze().numpy()
                class_label = class_id[j].item()
                fold = df.loc[i * train_dl.batch_size + j, 'fold']

                fold_folder = os.path.join(image_folder, f'fold{fold}')
                os.makedirs(fold_folder, exist_ok=True)

                file_name = f'image_{i}_{j}_{k}.png'

                plt.figure(figsize=(10, 8))
                plt.imshow(aug_sgram, aspect='auto', origin='lower')
                plt.xlabel('Time')
                plt.ylabel('Frequency')
                plt.title(f'Augmented Spectrogram (Channel: {k+1}, Class: {class_label})')
                plt.colorbar()

                save_path = os.path.join(fold_folder, file_name)
                plt.savefig(save_path)
                plt.close()
    print("Save success!")

if __name__ == '__main__':
    data_path = 'UrbanSound8K'
    metadata_file = data_path + '/metadata/UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df.head()

    myds = UrbanSoundDataset(df, data_path)

    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    img(train_dl)