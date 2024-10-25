from dataloader import load_dataset
from models import load_model

from utils import evaluate

import torch
from torch.utils.data import DataLoader
import argparse
import os, time, sys

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # GPU
    parser.add_argument("--use-cuda", action="store_true")
    
    # Model
    parser.add_argument("--model", type=str, default="MCT")
    
    # Dataset
    parser.add_argument("--dataset", default='EPHNOGRAM')
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=128)
    
    # Save
    parser.add_argument("--weights-filename", type=str)
    
    return parser

def print_setup(device, args):
    print("=======================[Settings]========================")
    print(f"\n  [GPU]")
    print(f"  |-[device]: {device}")
    print(f"\n  [MODEL]")
    print(f"  |-[model]: {args.model}")
    print(f"\n  [DATA]")
    print(f"  |-[dataset]: {args.dataset}")
    print(f"\n  [HYPERPARAMETERS]")
    print(f"  |-[batch size]: {args.batch_size}")
    print(f"\n [SAVE]")
    print(f"  |-[weights filename]: {args.weights_filename}")
    print("\n=======================================================")
    
    print("Proceed? [Y/N]: ", end="")
    proceed = input().lower()
    
    if proceed == 'n':
        sys.exit()

def main(args):
    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        
    print_setup(device, args)
    
    # Load Model
    model = load_model(args.model).to(device)
    ckpt = torch.load(os.path.join('saved/weights', args.weights_filename),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"It was trained {ckpt['epochs']} epochs")
    
    # Load Dataset(test)
    ds = load_dataset(dataset=args.dataset, mode='test')
    dl = DataLoader(ds,
                    shuffle=False,
                    batch_size=args.batch_size)
    
    start_time = int(time.time())
    results = evaluate(model, dl, device)
    elapsed_time = int(time.time() - start_time)
    print(f"Test Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s", end="\n\n")
    
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test ECG & PCG motor state classificiation model', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)