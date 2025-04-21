import torch
from torch.distributions.categorical import Categorical

import numpy as np
from model.transformer_dynamic import TransformerDynamics_2
# from model.transformer_dynamic_baseline_transformer import TransformerDynamics_2
# from model.transformer_dynamic_dit import TransformerDynamics_2
# from model.diffusion import EnVariationalDiffusion
from model.diffusion_2 import EnVariationalDiffusion_2
from egnn.egnn import EGNN
from egnn.models import EGNN_dynamics_QM9_MC
from egnn.egnn_vae import EGNN_VAE

# def get_model(args, device, dataset_info, dataloader_train):
#     histogram = dataset_info['n_nodes']
#     # histogram = {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
#     #             15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
#     #             8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 2}
#     # in_node_nf = 5 + (1 if include_charges else 0)
#     in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
#     nodes_dist = DistributionNodes(histogram)

#     prop_dist = None
#     if len(args.conditioning) > 0:
#         prop_dist = DistributionProperty(dataloader_train, args.conditioning)

#     if args.condition_time:
#         dynamics_in_node_nf = in_node_nf + 1
#     else:
#         print('Warning: dynamics model is _not_ conditioned on time.')
#         dynamics_in_node_nf = in_node_nf
    
#     #
#     #Only for Dit:
#     dynamics_in_node_nf = in_node_nf
#     print(f"Shape of dynamics_in_node_nf: {dynamics_in_node_nf}")
#     net_dynamics = TransformerDynamics(args=args,
#         in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
#         n_dims=3, device=device, hidden_nf=args.nf,
#         n_heads=args.n_heads,
#         n_layers=args.n_layers,
#         condition_time=args.condition_time
#         # in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
#         # n_dims=3, device=device, hidden_nf=args.nf,
#         # act_fn=torch.nn.SiLU(), 
#         # n_layers=args.n_layers,
#         # attention=args.attention, 
#         # tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
#         # inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
#         # normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method
#         )

#     # EGNN
#     # egnn = EGNN_dynamics_QM9(
#     #     in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
#     #     n_dims=3, device=device, hidden_nf=args.nf,
#     #     act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
#     #     attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
#     #     inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
#     #     normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)
#     egnn = EGNN_dynamics_QM9_MC(in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
#                  n_dims=3, device=device, hidden_nf=args.nf,
#                  act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
#                 #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
#                 #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'，
#                 num_vectors=7, num_vectors_out=3
#                  )
#     # egnn.eval()
    
#     if args.probabilistic_model == 'diffusion':
#         vdm = EnVariationalDiffusion(
#             dynamics=net_dynamics,
#             in_node_nf=in_node_nf,
#             n_dims=3,
#             timesteps=args.diffusion_steps,
#             noise_schedule=args.diffusion_noise_schedule,
#             noise_precision=args.diffusion_noise_precision,
#             loss_type=args.diffusion_loss_type,
#             norm_values=args.normalize_factors,
#             include_charges=args.include_charges,
#             egnn=egnn
#             )

#         return vdm, nodes_dist, prop_dist, egnn

#     else:
#         raise ValueError(args.probabilistic_model)

# def get_model_2(args, device, dataset_info, dataloader_train):
#     histogram = dataset_info['n_nodes']
#     # histogram = {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
#     #             15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
#     #             8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 2}
#     # in_node_nf = 5 + (1 if include_charges else 0)
#     in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
#     nodes_dist = DistributionNodes(histogram)

#     prop_dist = None
#     if len(args.conditioning) > 0:
#         prop_dist = DistributionProperty(dataloader_train, args.conditioning)

#     if args.condition_time:
#         dynamics_in_node_nf = in_node_nf + 1
#     else:
#         print('Warning: dynamics model is _not_ conditioned on time.')
#         dynamics_in_node_nf = in_node_nf
    
#     # print(f"Shape of dynamics_in_node_nf: {dynamics_in_node_nf}")
#     #For DiT
#     # dynamics_in_node_nf = in_node_nf
#     # print(f"dynamics_in_node_nf: {dynamics_in_node_nf}")
#     net_dynamics = TransformerDynamics_2(args=args,
#         in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
#         n_dims=3, device=device, hidden_nf=args.nf,
#         n_heads=args.n_heads,
#         n_layers=args.n_layers,
#         condition_time=args.condition_time
#         # in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
#         # n_dims=3, device=device, hidden_nf=args.nf,
#         # act_fn=torch.nn.SiLU(), 
#         # n_layers=args.n_layers,
#         # attention=args.attention, 
#         # tanh=args.tanh, mode=args.model, 
#         # norm_constant=args.norm_constant,
#         # inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
#         # normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method
#         )

#     # EGNN
#     # egnn = EGNN_dynamics_QM9(
#     #     in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
#     #     n_dims=3, device=device, hidden_nf=args.nf,
#     #     act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
#     #     attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
#     #     inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
#     #     normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)
#     # egnn = EGNN_dynamics_QM9_MC(in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
#     #              n_dims=3, device=device, hidden_nf=args.nf,
#     #              act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
#     #             #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
#     #             #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'，
#     #             num_vectors=7, num_vectors_out=3
#     #              )
#     # egnn.eval()
    
#     if args.probabilistic_model == 'diffusion':
#         vdm = EnVariationalDiffusion_2(
#             dynamics=net_dynamics,
#             in_node_nf=in_node_nf,
#             n_dims=3,
#             timesteps=args.diffusion_steps,
#             noise_schedule=args.diffusion_noise_schedule,
#             noise_precision=args.diffusion_noise_precision,
#             loss_type=args.diffusion_loss_type,
#             norm_values=args.normalize_factors,
#             include_charges=args.include_charges,
#             egnn=None
#             )

#         return vdm, nodes_dist, prop_dist

#     else:
#         raise ValueError(args.probabilistic_model)

def get_egnn_vae(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    # histogram = {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
    #             15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
    #             8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 2}
    # in_node_nf = 5 + (1 if include_charges else 0)
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf
    
    #EGNN
    egnn = EGNN_dynamics_QM9_MC(in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
                 n_dims=3, device=device, hidden_nf=args.nf,
                 act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
                #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'，
                num_vectors=7, num_vectors_out=2
                 )
    egnn_vae = EGNN_VAE(args=args,
        egnn=egnn,
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        condition_time=args.condition_time)
    
    return egnn_vae, nodes_dist, prop_dist

def get_diffusion(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    # histogram = {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
    #             15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
    #             8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 2}
    # in_node_nf = 5 + (1 if include_charges else 0)
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf
    
    #EGNN
    egnn = EGNN_dynamics_QM9_MC(in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
                 n_dims=3, device=device, hidden_nf=args.nf,
                 act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
                #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'，
                num_vectors=7, num_vectors_out=2
                 )
    if args.use_pretrain:
        pretrained_state_dict = torch.load(args.pretrained_model_path, map_location=device)
        # 提取EGNN相关参数
        if 'model_state_dict' in pretrained_state_dict:
            # 由于EGNN是作为组件传入EGNN_VAE的，所以参数前缀很可能是'egnn.'
            egnn_prefix = 'egnn.'
            
            # 提取EGNN参数
            egnn_state_dict = {}
            for key, value in pretrained_state_dict['model_state_dict'].items():
                if key.startswith(egnn_prefix):
                    # 去除前缀
                    new_key = key[len(egnn_prefix):]
                    egnn_state_dict[new_key] = value
            
            if len(egnn_state_dict) > 0:
                print(f"Found {len(egnn_state_dict)} EGNN parameters")
                
                # 尝试加载参数
                try:
                    missing_keys, unexpected_keys = egnn.load_state_dict(egnn_state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"Warning: Missing keys when loading EGNN: {missing_keys}")
                    if unexpected_keys:
                        print(f"Warning: Unexpected keys when loading EGNN: {unexpected_keys}")
                        
                    print("Successfully loaded pretrained EGNN parameters")
                except Exception as e:
                    print(f"Error loading EGNN parameters: {e}")
            else:
                print("No EGNN parameters found with prefix 'egnn.'")
                
                # 如果没找到，尝试打印所有键名以检查实际的前缀
                print("Available keys in model_state_dict:")
                for key in list(pretrained_state_dict['model_state_dict'].keys())[:20]:  # 只打印前20个键以避免过多输出
                    print(f"  {key}")
        else:
            print("pretrained_state_dict does not contain 'model_state_dict' key")
            print(f"Available keys: {pretrained_state_dict.keys()}")

    for param in egnn.parameters():
        param.requires_grad = False
    # print(f"Shape of dynamics_in_node_nf: {dynamics_in_node_nf}")
    #For DiT
    # dynamics_in_node_nf = in_node_nf
    # print(f"dynamics_in_node_nf: {dynamics_in_node_nf}")
    net_dynamics = TransformerDynamics_2(args=args,
        egnn=egnn,
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        condition_time=args.condition_time
        )
    
    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion_2(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges,
            egnn=None
            )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)

def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())