from .TestNet.testnet import TestNet
from .MCT.mct import MCT_Model
def load_model(model:str="MCT"):
    if model == "MCT":
        return MCT_Model()
    
    elif model == "TestNet":
        return TestNet()