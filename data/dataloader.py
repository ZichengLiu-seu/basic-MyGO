import os

import torch
from torch.utils.data import Dataset

from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd


def augment_sequence(augment_method, seq):
    seq_array = np.array(seq, dtype=np.float32)

    if augment_method == 'noise':
        noise = np.random.normal(0, 0.05, seq_array.shape)
        augmented_seq = seq_array + noise
    elif augment_method == 'time_warp':
        orig_time = np.arange(len(seq_array))
        new_time = orig_time + np.random.normal(0, 0.2, len(seq_array))
        new_time = np.clip(new_time, 0, len(seq_array) - 1)
        cs = CubicSpline(orig_time, seq_array, axis=0)
        augmented_seq = cs(new_time)
    elif augment_method == 'scaling':
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented_seq = seq_array * scale_factor
    else:
        augmented_seq = seq_array

    return augmented_seq.tolist()


class Locomotion_Dataset(Dataset):
    def __init__(self, root_path, flag='train', interaction_type='Touchpad', seq_len=60, pred_len=5):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.internal = int(self.seq_len / 4)
        self.interaction_type = interaction_type
        assert flag in ['train', 'val', 'test', 'all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.data_x = []
        self.label = []
        self.dirt = []
        self.motion_map = []

        num_list = os.listdir(os.path.join(self.root_path, 'motion'))
        for number in num_list:
            if number == '14':
                file_list = os.listdir(os.path.join(self.root_path, 'motion', number))
                for file_name in file_list:
                    if not file_name.endswith('ScenesmoothData.csv'):
                        continue
                    if not file_name.startswith(self.interaction_type) and self.interaction_type != 'All':
                        continue

                    df = pd.read_csv(os.path.join(self.root_path, 'motion', number, file_name))
                    num = int((len(df) - self.pred_len - self.seq_len) / self.internal) + 1
                    last_column = df.columns[-2]
                    other_column = df.columns[:-2]
                    label_column = df.columns[-4:-2]

                    dirt_raw = df.loc[:self.seq_len * num, last_column]
                    df_raw = df.loc[:self.seq_len * num, other_column]
                    label_raw = df.loc[:self.seq_len * num, label_column]

                    borders1 = [0, (int(num * 0.7) - 1) * self.internal, (int(num * 0.9) - 1) * self.internal, 0]
                    borders2 = [(int(num * 0.7) - 1) * self.internal + self.pred_len,
                                (int(num * 0.9) - 1) * self.internal + self.pred_len,
                                (num - 1) * self.internal + self.pred_len,
                                (int(num * 0.9) - 1) * self.internal + self.pred_len]
                    border1 = borders1[self.set_type]
                    border2 = borders2[self.set_type]

                    data = df_raw.values
                    label = label_raw.values
                    offset = len(self.data_x)
                    self.data_x.extend(data[border1:border2])
                    self.label.extend(label[border1:border2])
                    self.dirt.extend(dirt_raw.values[border1:border2])

                    start_num = len(self.motion_map)
                    motion_num = int((border2 - border1 - self.seq_len - self.pred_len) / self.internal) + 1
                    for index in range(motion_num):
                        self.motion_map.append(('original', index + start_num, offset, start_num))

    def __getitem__(self, index):
        data_type, i, offset, start_num = self.motion_map[index]
        s_begin = offset + int((i - start_num) * self.internal)
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        label = self.label[s_end + self.pred_len - 1]
        dirt = self.dirt[s_end + self.pred_len - 1]

        if data_type == 'augmented':
            seq_x = augment_sequence('noise', seq_x)

        return torch.tensor(np.array(seq_x, dtype=np.float32)), \
            torch.tensor(np.array(label, dtype=np.float32)), \
            torch.tensor(np.array(dirt, dtype=np.float32))

    def __len__(self):
        return len(self.motion_map)
