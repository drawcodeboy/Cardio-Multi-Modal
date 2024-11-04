from .metrics import get_metrics

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
import sys

def train_one_epoch(model, dataloader, optimizer, loss_fn, scheduler, device):
    model.train()
    
    total_loss = []
    for batch_idx, (x_ecg, x_pcg, targets) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        x_ecg = x_ecg.to(device)
        x_pcg = x_pcg.to(device)
        
        targets = targets.reshape(-1).to(device)
        
        logits = model(x_ecg, x_pcg)
        
        loss = loss_fn(logits, targets)
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.8f}", end="")
    print()
    
    scheduler.step(sum(total_loss)/len(total_loss))
    
    return sum(total_loss)/len(total_loss) # One Epoch Loss

@torch.no_grad()
def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    
    total_outputs = []
    total_targets = []
    
    for batch_idx, (x_ecg, x_pcg, targets) in enumerate(dataloader, start=1):
        x_ecg = x_ecg.to(device)
        x_pcg = x_pcg.to(device)
        
        targets = targets.reshape(-1)
        
        logits = model(x_ecg, x_pcg)
        # print(logits)
        # print(F.softmax(logits, dim=1))
        outputs = torch.argmax(F.softmax(logits, dim=1), dim=1)
        
        # print(outputs)
        # print(targets)
        
        total_outputs.extend(outputs.tolist())
        total_targets.extend(targets.tolist())
        
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    result = get_metrics(np.array(total_outputs), np.array(total_targets))
    
    return result
    
def inference_time():
    # Time 측정하는 함수
    pass