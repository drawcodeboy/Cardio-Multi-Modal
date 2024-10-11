from torch.utils.data import Dataset

import os
import wfdb
import pandas as pd

class ephnogram_dataloader(Dataset):
    def __init__(self,
                 dataset_path=r"data/physionet.org/files/ephnogram/1.0.0",
                 mode='train',
                 seq_seconds:int=5, #seconds
                 transform=None):
        super().__init__()
        
        # (1) Initialization
        if not os.path.isdir(dataset_path):
            raise ValueError(f"check [dataset_path] arg, your [dataset_path] is {dataset_path}")
        self.dataset_path = dataset_path
        
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Check Dataset [mode] arg, your [mode] is {mode}")
        self.mode = mode
        
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
        
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        sample_path, start_point, label = self.data_li[idx]
        
        sample = wfdb.rdsamp(sample_path)[0][start_point:start_point+self.sample_len]
        x_ecg, x_pcg = sample[:, 0], sample[:, 1]
        
        if start_point == 0: # Issue: PCG는 start point가 항상 1임 -> t_0 -> 0으로 초기화
            x_pcg[0] = 0.
            
        # Implementation List
        # (1) Downsampling
        # (2) transform for preprocessing (Normalization)
        
        return x_ecg, x_pcg, label
    
    def _check(self):
        metadata = pd.read_csv(f"{self.dataset_path}/ECGPCGSpreadsheet.csv")
        
        for i in range(metadata.shape[0]):
            if i == 2: break # for debugging
            print(f"\rLoad Dataset[{i+1:02d}/{metadata.shape[0]:02d}] ({100*(i+1)/metadata.shape[0]:.2f}%)", end="")
            
            row = metadata.iloc[i]
            
            # Except NaN
            if row[['Record Name', 'Record Duration (min)', 'Recording Scenario']].isnull().sum() >= 1:
                continue
            
            # Read Sample
            sample_path = f"{self.dataset_path}/WFDB/{row['Record Name']}"
            try:
                sample = wfdb.rdsamp(sample_path)
            except:
                continue
            
            # Set Label
            label = self.class_map[row['Recording Scenario']]
            
            # Add to self.data_li
            for start_point in range(0, sample[1]['sig_len'], self.sample_len):
                self.data_li.append([sample_path, start_point, label])
        print()

if __name__ == '__main__':
    ds = ephnogram_dataloader()
    x_ecg, x_pcg, label = ds[362]
    
    print(x_ecg.shape, x_pcg.shape)
    print(len(ds))
    
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