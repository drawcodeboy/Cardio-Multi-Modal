from .TestNet.testnet import TestNet

def load_model(model:str="MCT"):
    if model == "MCT":
        return None
    
    elif model == "TestNet":
        return TestNet()