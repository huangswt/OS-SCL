import glob
import itertools
import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


class Wave2Mel(object):
    def __init__(self, sr=16000,
                 n_fft=2048,
                 n_mels=128,
                 win_length=2048,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power,
                                                                  pad_mode='constant',
                                                                  norm='slaney',
                                                                  mel_scale='slaney'
                                                                  )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, test_name, name_list, cfg):
        dataset_dir = os.path.join(root_path, test_name, 'test')
        normal_files = sorted(glob.glob('{dir}/normal_*'.format(dir=dataset_dir)))
        anomaly_files = sorted(glob.glob('{dir}/anomaly_*'.format(dir=dataset_dir)))

        self.test_files = np.concatenate((normal_files, anomaly_files), axis=0)
        self.cfg = cfg

        self.Logmel = Wave2Mel()

        normal_labels = np.zeros(len(normal_files))
        anomaly_labels = np.ones(len(anomaly_files))
        self.y_true = torch.LongTensor(np.concatenate((normal_labels, anomaly_labels), axis=0))

        target_idx = name_list.index(test_name)

        label_init_num = 0
        for i, name in enumerate(name_list):
            if i == target_idx:
                break
            label_init_num += len(self._get_label_list(name))

        self.labels = []
        self.id_labels = []
        label_list = self._get_label_list(test_name)
        for file_name in self.test_files:
            for idx, label_idx in enumerate(label_list):
                if label_idx in file_name:
                    self.labels.append(idx + label_init_num)
                    self.id_labels.append(label_idx)

        self.labels = torch.LongTensor(self.labels)

        self.y_list = []
        self.y_spec_list = []

        for i in tqdm(range(len(self.test_files)), desc=f'{test_name} Log-Mel Convetor', dynamic_ncols=True):
            y, sr = self._file_load(self.test_files[i])
            y_specgram = self.Logmel(y)
            self.y_list.append(y)
            self.y_spec_list.append(y_specgram)

    def __getitem__(self, idx):
        anomal_label = self.y_true[idx]
        label = self.labels[idx]
        id_label = self.id_labels[idx]
        return self.y_list[idx], self.y_spec_list[idx], label, anomal_label, id_label

    def __len__(self):
        return len(self.test_files)

    def _file_load(self, file_name):
        try:
            y, sr = torchaudio.load(file_name)
            y = y[..., :sr * 10]
            return y, sr
        except:
            print("file_broken or not exists!! : {}".format(file_name))

    def _get_label_list(self, name):
        if name == 'ToyConveyor':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']

        elif name == 'ToyCar':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07']

        else:
            label_list = ['id_00', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']

        return label_list


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, name_list, cfg):
        data_path = [os.path.join(root_path, name, 'train') for name in name_list]
        files_list = [self._file_list_generator(target_path) for target_path in data_path]

        self.labels = []
        self.cfg = cfg

        self.Logmel = Wave2Mel()

        maximum = 0
        for i, files in enumerate(files_list):
            label_list = self._get_label_list(name_list[i])
            for file_name in files:
                for idx, label_idx in enumerate(label_list):
                    if label_idx in file_name:
                        self.labels.append(idx + maximum)
            maximum = max(self.labels) + 1

        self.unrolled_files_list = list(itertools.chain.from_iterable(files_list))

        unique_labels = set(self.labels)

        num_unique_labels = len(unique_labels)

        print(f"class :{num_unique_labels}")

        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, idx):

        y, sr = self._file_load(self.unrolled_files_list[idx])
        y_specgram = self.Logmel(y)
        return y, y_specgram, self.labels[idx]

    def __len__(self):
        return len(self.unrolled_files_list)

    def _get_label_list(self, name):
        if name == 'ToyConveyor':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']

        elif name == 'ToyCar':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07']

        else:
            label_list = ['id_00', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']

        return label_list

    def _file_list_generator(self, target_dir):
        training_list_path = os.path.abspath('{dir}/*.wav'.format(dir=target_dir))
        files = sorted(glob.glob(training_list_path))
        if len(files) == 0:
            print('no_wav_file!!')
        return files

    def _file_load(self, file_name):
        try:
            y, sr = torchaudio.load(file_name)
            y = y[..., :sr * 10]
            return y, sr
        except:
            print("file_broken or not exists!! : {}".format(file_name))
