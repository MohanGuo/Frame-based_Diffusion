import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
from types import SimpleNamespace

# Assuming these are from your project
from equivariant_diffusion.utils import remove_mean_with_mask, assert_mean_zero_with_mask
from equivariant_diffusion.utils import sample_center_gravity_zero_gaussian_with_mask
import utils
import qm9.utils as qm9utils
import qm9.visualizer as vis

dtype = torch.float32

def train_egnn_vae(args, loader, epoch, model, device, optimizer, beta=0.1, dataset_info=None):
    """
    Training function for EGNN_VAE model
    
    Args:
        args: Configuration arguments
        loader: Data loader
        epoch: Current epoch number
        model: EGNN_VAE model instance
        device: Computing device
        optimizer: Model optimizer
        beta: KL divergence weight
    """
    model.train()
    loss_epoch = []
    recon_loss_epoch = []
    kl_loss_epoch = []
    
    n_iterations = len(loader)
    dtype = torch.float32
    
    for i, data in enumerate(loader):
        # Extract data
        x = data['positions'].to(device)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device)
        
        # Center the molecule
        x = remove_mean_with_mask(x, node_mask)
        
        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise
            x = remove_mean_with_mask(x, node_mask)
        
        # Optional data augmentation with random rotation
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()
        
        # Prepare input features
        if args.include_charges:
            h_features = torch.cat([one_hot, charges], dim=-1)
        else:
            h_features = one_hot
        
        # Combine coordinates and features
        xh = torch.cat([x, h_features], dim=2)
        
        # Prepare context if using conditioning
        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device)
        else:
            context = None
        
        # Forward pass
        optimizer.zero_grad()
        model_output, loss_dict = model._forward(xh, node_mask, edge_mask, context)
        
        # Get losses
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if enabled
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Log progress
        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss: {loss.item():.4f}, Recon: {loss_dict['recon_loss'].item():.4f}, "
                  f"KL: {loss_dict['kl_loss'].item():.4f}")
        
        # Collect statistics
        loss_epoch.append(loss.item())
        recon_loss_epoch.append(loss_dict['recon_loss'].item())
        kl_loss_epoch.append(loss_dict['kl_loss'].item())
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                "Batch Loss": loss.item(), 
                "Batch Recon Loss": loss_dict['recon_loss'].item(),
                "Batch KL Loss": loss_dict['kl_loss'].item(),
                "Batch X Recon Loss": loss_dict['x_recon_loss'].item(),
                "Batch H Recon Loss": loss_dict['h_recon_loss'].item()
            }, commit=True)
        
        # Visualize and save samples periodically
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            visualize_reconstructions(model, xh, node_mask, edge_mask, context, args, dataset_info, epoch, batch_id=str(i))
        
        # Debug option to break early
        if args.break_train_epoch:
            break
    
    # Log epoch statistics
    avg_loss = np.mean(loss_epoch)
    avg_recon_loss = np.mean(recon_loss_epoch)
    avg_kl_loss = np.mean(kl_loss_epoch)
    
    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon_loss:.4f}, Avg KL: {avg_kl_loss:.4f}")
    
    if not args.no_wandb:
        wandb.log({
            "Train Epoch Loss": avg_loss,
            "Train Epoch Recon Loss": avg_recon_loss,
            "Train Epoch KL Loss": avg_kl_loss
        }, commit=False)
    
    return avg_loss


def test_egnn_vae(args, loader, epoch, model, device, partition='Test'):
    """
    Testing function for EGNN_VAE model
    
    Args:
        args: Configuration arguments
        loader: Data loader
        epoch: Current epoch number
        model: EGNN_VAE model instance
        device: Computing device
        partition: Data partition name ('Test', 'Validation', etc.)
    """
    model.eval()
    with torch.no_grad():
        loss_epoch = []
        recon_loss_epoch = []
        kl_loss_epoch = []
        
        n_iterations = len(loader)
        
        for i, data in enumerate(loader):
            # Extract data - same as in training
            x = data['positions'].to(device)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            # Center the molecule
            x = remove_mean_with_mask(x, node_mask)
            
            # Prepare input features
            if args.include_charges:
                h_features = torch.cat([one_hot, charges], dim=-1)
            else:
                h_features = one_hot
            
            # Combine coordinates and features
            xh = torch.cat([x, h_features], dim=2)
            
            # Prepare context if using conditioning
            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device)
            else:
                context = None
            
            # Forward pass
            model_output, loss_dict = model._forward(xh, node_mask, edge_mask, context)
            
            # Get losses
            loss = loss_dict['loss']
            
            # Collect statistics
            loss_epoch.append(loss.item())
            recon_loss_epoch.append(loss_dict['recon_loss'].item())
            kl_loss_epoch.append(loss_dict['kl_loss'].item())
            
            # Log progress
            if i % args.n_report_steps == 0:
                print(f"\r{partition} Loss \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"Loss: {np.mean(loss_epoch):.4f}")
            
            # Debug option to break early
            if args.break_train_epoch:
                break
        
        # Calculate average metrics
        avg_loss = np.mean(loss_epoch)
        avg_recon_loss = np.mean(recon_loss_epoch)
        avg_kl_loss = np.mean(kl_loss_epoch)
        
        print(f"{partition} Epoch {epoch} - Avg Loss: {avg_loss:.4f}, "
              f"Avg Recon: {avg_recon_loss:.4f}, Avg KL: {avg_kl_loss:.4f}")
        
        if not args.no_wandb:
            wandb.log({
                f"{partition} Epoch Loss": avg_loss,
                f"{partition} Epoch Recon Loss": avg_recon_loss,
                f"{partition} Epoch KL Loss": avg_kl_loss
            }, commit=True)
        
        return avg_loss


def visualize_reconstructions(model, xh, node_mask, edge_mask, context, args, dataset_info, epoch, batch_id=''):
    """
    Visualize reconstructions from the VAE model, comparing original molecules
    with their mean reconstructions.
    
    Args:
        model: EGNN_VAE model
        xh: Input data (positions and features)
        node_mask: Node mask
        edge_mask: Edge mask
        context: Conditioning context
        args: Configuration arguments
        dataset_info: Dataset information
        epoch: Current epoch
        batch_id: Batch identifier
    """
    # Set model to eval mode for visualization
    model.eval()
    with torch.no_grad():
        # Forward pass with mean encoding (no randomness)
        model_output, loss_dict = model._forward(xh, node_mask, edge_mask, context, use_mean=True)
        
        # Get original data
        bs, n_nodes, dims = xh.shape
        x_original = xh[:, :, :3]
        
        # Create save directories
        save_dir = f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/reconstructions/'
        os.makedirs(f'{save_dir}/original/', exist_ok=True)
        os.makedirs(f'{save_dir}/reconstructed/', exist_ok=True)
        
        # Number of molecules to visualize
        n_vis = min(5, bs)
        
        # Determine atom type dimension from dataset_info
        atom_type_dim = len(dataset_info['atom_decoder'])
        
        # Save original and reconstructed molecules
        for i in range(n_vis):
            # Original molecule
            orig_x = x_original[i:i+1]
            orig_h = xh[i:i+1, :, 3:]
            
            # Extract one-hot and charges for original
            orig_one_hot = orig_h[:, :, :atom_type_dim]
            if args.include_charges:
                orig_charges = orig_h[:, :, atom_type_dim:]
            else:
                orig_charges = torch.zeros((orig_h.size(0), orig_h.size(1), 0), device=orig_x.device)
            
            # Save original
            vis.save_xyz_file(f'{save_dir}/original/', 
                             orig_one_hot, orig_charges, orig_x, dataset_info, i, name=f'molecule_{i}')
            
            # Reconstructed molecule
            recon_x = model_output[i:i+1, :, :3]
            recon_h = model_output[i:i+1, :, 3:]
            
            # Extract one-hot and charges for reconstruction
            recon_one_hot = recon_h[:, :, :atom_type_dim]
            if args.include_charges:
                recon_charges = recon_h[:, :, atom_type_dim:]
            else:
                recon_charges = torch.zeros((recon_h.size(0), recon_h.size(1), 0), device=recon_x.device)
            
            # Save reconstruction
            vis.save_xyz_file(f'{save_dir}/reconstructed/', 
                             recon_one_hot, recon_charges, recon_x, dataset_info, i, name=f'molecule_{i}')
        
        # Log reconstruction quality metrics
        if 'recon_loss' in loss_dict:
            print(f"Epoch {epoch} Batch {batch_id} - Reconstruction Loss: {loss_dict['recon_loss'].item():.4f}")
        if 'kl_loss' in loss_dict:
            print(f"Epoch {epoch} Batch {batch_id} - KL Loss: {loss_dict['kl_loss'].item():.4f}")
    
    # Set model back to training mode
    model.train()