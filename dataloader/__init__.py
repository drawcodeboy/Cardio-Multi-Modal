from .ephnogram_dataloader import ephnogram_dataloader
from .temp_dataloader import temp_dataloader

def load_dataset(dataset:str="EPHNOGRAM",
                 mode:str='train'):
    
    if dataset == "EPHNOGRAM":
        return ephnogram_dataloader(mode=mode)
    
    elif dataset == "Temp":
        return temp_dataloader(mode=mode)