import numpy as np
import getpass
import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Folders
def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass


# Model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


# def gradient_clipping(flow, gradnorm_queue):
#     # Allow gradient norm to be 150% + 2 * stdev of the recent history.
#     max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

#     # Clips gradient and returns the norm
#     grad_norm = torch.nn.utils.clip_grad_norm_(
#         flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

#     if float(grad_norm) > max_grad_norm:
#         gradnorm_queue.add(float(max_grad_norm))
#     else:
#         gradnorm_queue.add(float(grad_norm))

#     if float(grad_norm) > max_grad_norm:
#         print(f'Clipped gradient with value {grad_norm:.1f} '
#               f'while allowed {max_grad_norm:.1f}')
#     return grad_norm

def gradient_clipping(args, flow, gradnorm_queue, clipping_type="queue"):

    if clipping_type == "queue":
        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

        # Clips gradient and returns the norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

        if float(grad_norm) > max_grad_norm:
            gradnorm_queue.add(float(max_grad_norm))
        else:
            gradnorm_queue.add(float(grad_norm))

        return grad_norm

    elif clipping_type == "norm":
        grad_norm = torch.nn.utils.clip_grad_norm_(
            flow.parameters(), max_norm=args.max_grad_norm, norm_type=2.0)
        gradnorm_queue.add(float(grad_norm))
        return grad_norm

    else:
        raise ValueError

# Rotation data augmntation
# def random_rotation(x):
#     bs, n_nodes, n_dims = x.size()
#     device = x.device
#     angle_range = np.pi * 2
#     if n_dims == 2:
#         theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
#         cos_theta = torch.cos(theta)
#         sin_theta = torch.sin(theta)
#         R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
#         R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
#         R = torch.cat([R_row0, R_row1], dim=1)

#         x = x.transpose(1, 2)
#         x = torch.matmul(R, x)
#         x = x.transpose(1, 2)

#     elif n_dims == 3:

#         # Build Rx
#         Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
#         theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
#         cos = torch.cos(theta)
#         sin = torch.sin(theta)
#         Rx[:, 1:2, 1:2] = cos
#         Rx[:, 1:2, 2:3] = sin
#         Rx[:, 2:3, 1:2] = - sin
#         Rx[:, 2:3, 2:3] = cos

#         # Build Ry
#         Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
#         theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
#         cos = torch.cos(theta)
#         sin = torch.sin(theta)
#         Ry[:, 0:1, 0:1] = cos
#         Ry[:, 0:1, 2:3] = -sin
#         Ry[:, 2:3, 0:1] = sin
#         Ry[:, 2:3, 2:3] = cos

#         # Build Rz
#         Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
#         theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
#         cos = torch.cos(theta)
#         sin = torch.sin(theta)
#         Rz[:, 0:1, 0:1] = cos
#         Rz[:, 0:1, 1:2] = sin
#         Rz[:, 1:2, 0:1] = -sin
#         Rz[:, 1:2, 1:2] = cos

#         x = x.transpose(1, 2)
#         x = torch.matmul(Rx, x)
#         #x = torch.matmul(Rx.transpose(1, 2), x)
#         x = torch.matmul(Ry, x)
#         #x = torch.matmul(Ry.transpose(1, 2), x)
#         x = torch.matmul(Rz, x)
#         #x = torch.matmul(Rz.transpose(1, 2), x)
#         x = x.transpose(1, 2)
#     else:
#         raise Exception("Not implemented Error")

#     return x.contiguous()

def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2

    if n_dims == 3:
        axis = torch.randn(bs, 3, device=device)
        axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
        angles = torch.rand(bs, 1, device=device) * angle_range - np.pi
        
        K = torch.zeros(bs, 3, 3, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        I = torch.eye(3, device=device).unsqueeze(0).repeat(bs, 1, 1)
        sin_theta = torch.sin(angles).unsqueeze(-1)
        cos_theta = (1 - torch.cos(angles)).unsqueeze(-1)
        R = I + sin_theta * K + cos_theta * (K @ K)
        
        x = x.transpose(1, 2)          # [bs, 3, n_nodes]
        x = torch.bmm(R, x)            # 
        x = x.transpose(1, 2)          # [bs, n_nodes, 3]
    else:
        # 2D  
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)
    
    return x.contiguous()

def visualize_molecule(x, node_mask, batch_index=0, save_path='molecule.png'):
    coords = x[batch_index].detach().cpu().numpy()
    mask = node_mask[batch_index].squeeze().detach().cpu().numpy()
    valid_mask = mask > 0
    valid_coords = coords[valid_mask]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(
        valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2],
        c=valid_coords[:, 2], cmap='viridis', s=100, depthshade=True,
        edgecolors='k', linewidth=0.5
    )
    
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Z Coordinate', rotation=270, labelpad=15)
    
    max_range = np.array([
        valid_coords[:,0].max()-valid_coords[:,0].min(), 
        valid_coords[:,1].max()-valid_coords[:,1].min(),
        valid_coords[:,2].max()-valid_coords[:,2].min()
    ]).max() * 0.5
    
    mid_x = (valid_coords[:,0].max()+valid_coords[:,0].min()) * 0.5
    mid_y = (valid_coords[:,1].max()+valid_coords[:,1].min()) * 0.5
    mid_z = (valid_coords[:,2].max()+valid_coords[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=30, azim=45)
    
    for i in range(len(valid_coords)):
        for j in range(i+1, len(valid_coords)):
            ax.plot(
                [valid_coords[i,0], valid_coords[j,0]],
                [valid_coords[i,1], valid_coords[j,1]], 
                [valid_coords[i,2], valid_coords[j,2]],
                'gray', alpha=0.3, linewidth=0.5
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Other utilities
def get_wandb_username(username):
    if username == 'cvignac':
        return 'cvignac'
    current_user = getpass.getuser()
    if current_user == 'victor' or current_user == 'garciasa':
        return 'vgsatorras'
    else:
        return username


if __name__ == "__main__":


    ## Test random_rotation
    bs = 2
    n_nodes = 16
    n_dims = 3
    x = torch.randn(bs, n_nodes, n_dims)
    print(x)
    x = random_rotation(x)
    #print(x)
