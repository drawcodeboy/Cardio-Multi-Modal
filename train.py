from dataloader import load_dataset
from models import load_model

from utils import train_one_epoch, save_model_ckpt, save_loss_ckpt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse, time, os

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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    
    # Save
    parser.add_argument("--save-weights-dir", type=str, default="saved/weights")
    parser.add_argument("--save-losses-dir", type=str, default="saved/losses")
    
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
    print(f"  |-[epochs]: {args.epochs}")
    print(f"  |-[lr]: {args.lr:06f}")
    print(f"  |-[batch size]: {args.batch_size}")
    print(f"\n [SAVE]")
    print(f"  |-[SAVE WEIGHTS DIR]: {args.save_weights_dir}")
    print(f"  |-[SAVE LOSSES DIR]: {args.save_losses_dir}")
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
    
    # Load Dataset(train)
    ds = load_dataset(dataset=args.dataset, mode='train')
    dl = DataLoader(ds,
                    shuffle=True,
                    batch_size=args.batch_size)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Scheduler (논문에는 쓴다고 나와있지는 않았으나, 쓰는 것도 좋을 듯)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-7)
    
    # Training
    total_train_loss = []
    
    for current_epoch in range(0, args.epochs):
        current_epoch += 1
        print("=======================================================")
        print(f"Epoch: [{current_epoch:03d}/{args.epochs:03d}]", end="\n\n")
        
        start_time = int(time.time())
        train_loss = train_one_epoch(model, dl, optimizer, loss_fn, scheduler, device)
        elapsed_time = int(time.time() - start_time)
        print(f"Training Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s", end="\n\n")
        total_train_loss.append(train_loss)
        
    save_model_ckpt(model, args.model, args.epochs, args.save_weights_dir)
    save_loss_ckpt(args.model, args.epochs, total_train_loss, args.save_losses_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training ECG & PCG motor state classificiation model', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)