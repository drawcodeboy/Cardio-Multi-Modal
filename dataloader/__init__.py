from .ephnogram_dataloader import ephnogram_dataloader

def load_dataset(dataset:str="EPHNOGRAM"):
    
    if dataset == "EPHNOGRAM":
        return ephnogram_dataloader()