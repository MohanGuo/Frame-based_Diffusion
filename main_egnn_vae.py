try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from models import get_egnn_vae, get_optim
from equivariant_diffusion.utils import assert_correctly_masked, remove_mean_with_mask
from equivariant_diffusion.utils import sample_center_gravity_zero_gaussian_with_mask
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
import numpy as np
import os
from qm9.utils import prepare_context, compute_mean_mad
from types import SimpleNamespace
import qm9.visualizer as vis
from train_test_egnn_vae import train_egnn_vae, test_egnn_vae, visualize_reconstructions

# Parser arguments
parser = argparse.ArgumentParser(description='EGNN_VAE Training')
parser.add_argument('--exp_name', type=str, default='egnn_vae')
parser.add_argument('--model', type=str, default='egnn_vae',
                    help='egnn_vae')

# Training params
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta', type=float, default=0.1, 
                    help='KL divergence weight in VAE loss')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--max_grad_norm', type=float, default=10.0,
                    help='Maximum gradient norm for clipping')

# EGNN args
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of EGNN layers')
parser.add_argument('--nf', type=int, default=128,
                    help='hidden dimension size')
parser.add_argument('--latent_dim', type=int, default=8, 
                    help='VAE latent dimension size')
parser.add_argument('--n_dims', type=int, default=3, 
                    help='number of coordinate dimensions')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--coords_range', type=float, default=15.0,
                    help='Coordinate range')
parser.add_argument('--agg', type=str, default='sum',
                    help='"sum" or "mean"')
parser.add_argument('--condition_time', type=eval, default=False,
                    help='True | False')

# Dataset args
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--data_augmentation', type=eval, default=False, 
                    help='use random rotations for data augmentation')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--augment_noise', type=float, default=0)

# Logging and saving args
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--save_epochs', type=int, default=10,
                    help='save model every n epochs')
parser.add_argument('--num_workers', type=int, default=0, 
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--resume', type=str, default=None,
                    help='Resume from checkpoint')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Starting epoch for resuming')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")

args = parser.parse_args()

# Get dataset information
dataset_info = get_dataset_info(args.dataset, args.remove_h)

# Set up CUDA
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float

# Set up directories
utils.create_folders(args)

# Set up wandb
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {
    'entity': utils.get_wandb_username(args.wandb_usr), 
    'name': args.exp_name, 
    'project': 'egnn-vae', 
    'config': args,
    'settings': wandb.Settings(_disable_stats=True), 
    'reinit': True, 
    'mode': mode
}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

# Process conditioning information
if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    data_dummy = next(iter(dataloaders['train']))
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
    args.context_node_nf = context_node_nf
else:
    context_node_nf = 0
    property_norms = None
    args.context_node_nf = 0


# Main function
def main():
    # Get EGNN_VAE model
    model, nodes_dist, prop_dist = get_egnn_vae(args, device, dataset_info, dataloaders['train'])
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create save directory
    os.makedirs(f'outputs/{args.exp_name}/checkpoints', exist_ok=True)
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, args.n_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_egnn_vae(
            args=args,
            loader=dataloaders['train'],
            epoch=epoch,
            model=model,
            device=device,
            optimizer=optimizer,
            beta=args.beta,
            dataset_info=dataset_info
        )
        
        print(f"Epoch {epoch} took {time.time() - start_time:.1f} seconds")
        
        # Test periodically
        if epoch % args.test_epochs == 0:
            # Validation
            val_loss = test_egnn_vae(
                args=args,
                loader=dataloaders['valid'],
                epoch=epoch,
                model=model,
                device=device,
                partition='Val'
            )
            
            # Test
            test_loss = test_egnn_vae(
                args=args,
                loader=dataloaders['test'],
                epoch=epoch,
                model=model,
                device=device,
                partition='Test'
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save_model:
                    print(f"Saving best model with validation loss: {val_loss:.4f}")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                    }, f'outputs/{args.exp_name}/checkpoints/best_model.pt')
                    
            # Save periodic checkpoint
            if epoch % args.save_epochs == 0 and args.save_model:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'test_loss': test_loss,
                }, f'outputs/{args.exp_name}/checkpoints/checkpoint_epoch{epoch}.pt')

    # Save final model
    if args.save_model:
        torch.save({
            'epoch': args.n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'outputs/{args.exp_name}/checkpoints/final_model.pt')
    
    print("Training completed!")

if __name__ == "__main__":
    main()