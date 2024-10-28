import torch
from torch.utils.data import Dataset

import os
import wfdb
import pandas as pd
import random
from scipy.signal import decimate

class ephnogram_dataloader(Dataset):
    def __init__(self,
                dataset_path=r"data/physionet.org/files/ephnogram/1.0.0",
                mode='train',
                split_ratio:float=0.7,
                seq_seconds:int=5, #seconds
                transform=None,
                check_load:bool=False):
        super().__init__()
        
        # (1) Initialization
        if not os.path.isdir(dataset_path):
            raise ValueError(f"check [dataset_path] arg, your [dataset_path] is {dataset_path}")
        self.dataset_path = dataset_path
        
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Check Dataset [mode] arg, your [mode] is {mode}")
        self.mode = mode
        
        self.split_ratio = split_ratio
        
        self.check_load = check_load
        
        self.seq_seconds = seq_seconds
        self.fs = 8000
        self.sample_len = self.seq_seconds * self.fs
        
        # Class Mapping (7 labels)
        self.class_map = {
            "Exercise: pedaling a stationary bicycle": 0,
            "Rest: sitting on armchair": 1,
            "Rest: laying on bed": 2,
            "Exercise: slow walk (7 min); fast walk (8 min); sit down and stand up (4 min); slow walk (6 min);rest": 3,
            "Exercise: slow walk (7 min); fast walk (8 min); sit down and stand up (4 min); slow walk (6 min); rest": 3,
            "Exercise: Bruce protocol treadmill stress test": 4,
            "Exercise: walking at constant speed (3.7 km/h) ": 5,
            "Exercise: bicycle stress test": 6,
        }
        
        # (2) Check & Get Sample path and Label
        self.data_li = [] # element is list that contains [sample_path, start_point, label]
        self._check()
        
        # (3) Randomly Sample, fixed seed = 42
        random.seed(42)
        random.shuffle(self.data_li)
        
        # (4) mode
        if self.mode == 'train':
            self.data_li = self.data_li[:int(self.split_ratio*len(self.data_li))]
        elif self.mode == 'test':
            self.data_li = self.data_li[int(self.split_ratio*len(self.data_li)):]
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        sample_path, start_point, label = self.data_li[idx]
        
        # sample = wfdb.rdsamp(sample_path)[0][start_point:start_point+self.sample_len]
        # 위 코드는 느림.
        sample = wfdb.rdsamp(sample_path, 
                             sampfrom=start_point, 
                             sampto=start_point+self.sample_len)[0]
        x_ecg, x_pcg = sample[:, 0], sample[:, 1]
        
        if start_point == 0: # Issue: PCG는 start point가 항상 1임 -> t_0 -> 0으로 초기화
            x_pcg[0] = 0.
            
        # Downsampling (8000Hz -> 1000Hz)
        x_ecg = decimate(x_ecg, 8)
        x_pcg = decimate(x_pcg, 8)
        
        if self.transform is not None:
            x_ecg, x_pcg = self.transform(x_ecg, x_pcg)
        
        # To Tensor
        x_ecg = torch.tensor(x_ecg.copy(), dtype=torch.float32).reshape(-1, 1)
        x_pcg = torch.tensor(x_pcg.copy(), dtype=torch.float32).reshape(-1, 1)
        label = torch.tensor([label], dtype=torch.int64) # long type for label
        
        return x_ecg, x_pcg, label
    
    def _check(self):
        # label_cnt = [0 for i in range(0, 7)] # for debugging
        metadata = pd.read_csv(f"{self.dataset_path}/ECGPCGSpreadsheet.csv")
        
        for i in range(metadata.shape[0]):
            # if i == 1: break # for debugging
            print(f"\rLoad Dataset[{i+1:02d}/{metadata.shape[0]:02d}] ({100*(i+1)/metadata.shape[0]:.2f}%)", end="")
            
            row = metadata.iloc[i]
            
            # Except NaN
            if row[['Record Name', 'Record Duration (min)', 'Recording Scenario']].isnull().sum() >= 1:
                continue
            
            # Read Sample
            sample_path = f"{self.dataset_path}/WFDB/{row['Record Name']}"
            
            if self.check_load == True:
                try:
                    sample = wfdb.rdsamp(sample_path)
                except:
                    continue
            
            # Set Label
            label = self.class_map[row['Recording Scenario']]
            
            # Set Sample Total Length
            sample_total_length = 30*8000 if row['Record Duration (min)'] == 0.5 else 30*60*8000
            
            # Add to self.data_li
            for start_point in range(0, sample_total_length, self.sample_len):
                self.data_li.append([sample_path, start_point, label])
            
            # label_cnt[label] += (sample[0].shape[0] // self.sample_len) # for debugging
        
        # print(label_cnt) # for debugging
        print()

if __name__ == '__main__':
    train_ds = ephnogram_dataloader(mode='train')
    test_ds = ephnogram_dataloader(mode='test')
    
    # for i, (x1, x2, label) in enumerate(train_ds):
    #     print(f"\rload {i}", end='')
    
    x_ecg, x_pcg, label = train_ds[0]
    
    print(x_ecg.shape, x_pcg.shape)
    print(len(train_ds)), print(len(test_ds)) # if 5secs sampling, number of samples is 22,008
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.arange(0, x_ecg.shape[0], 1)
    
    fig, axes = plt.subplots(2, 1)
    axes[0].set_title("ECG")
    axes[0].plot(x, x_ecg, color='r')
    axes[1].set_title("PCG")
    axes[1].plot(x, x_pcg, color='r')
    
    plt.tight_layout()
    plt.show()