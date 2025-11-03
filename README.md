# 🧬 Frame-based Equivariant Diffusion Models for 3D Molecular Generation

Official implementation of the paper:  

📄 **[Frame-based Equivariant Diffusion Models for 3D Molecular Generation]([https://arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/2509.19506))** 

*(arXiv preprint, 2025)*

**If** you want to set-up a rdkit environment, it may be easiest to install conda and run:
``conda create -c conda-forge -n my-rdkit-env rdkit``

and then install the other required packages from there. The code should still run without rdkit installed though.

## 🧠 Overview

This repository provides implementations of **Frame-based Diffusion Models** for molecular generation,  
including three coordinated variants developed during the research:

| Branch | Variant | Description |
|:-------|:---------|:-------------|
| **main** | **GFD** | Global Frame Diffusion Model |
| **LFD** | Local Frame Diffusion | Local frame-based model variant |
| **IFD** | Invariant Frame Diffusion | Invariant frame-based model variant |

Each variant can be accessed by switching to its corresponding branch with different backbone selections.

## GFD

### Train GFD with EdgeDiT on QM9
To train GFD with EdgeDiT:

```bash
python main_qm9.py \
--n_epochs 5000 \
--exp_name ${EXP_NAME} \
--n_stability_samples 1000 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 \
--diffusion_steps 1000 \
--diffusion_loss_type l2 \
--batch_size 256 \
--nf 256 \
--n_layers 9 \
--lr 2e-4 \
--normalize_factors [1,4,10] \
--test_epochs 20 \
--ema_decay 0.9999 \
--datadir $DATA_DIR \
--inte_model transformer_dit \
--xh_hidden_size 184 --K 184 \
--hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4.0 --mlp_dropout 0.0 \
```

To train GFD* with EdgeDiT:

```bash
python main_qm9.py \
--n_epochs 5000 \
--exp_name ${EXP_NAME} \
--n_stability_samples 1000 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 \
--diffusion_steps 1000 \
--diffusion_loss_type l2 \
--batch_size 256 \
--nf 256 \
--n_layers 9 \
--lr 1e-4 \
--normalize_factors [1,4,10] \
--test_epochs 30 \
--ema_decay 0.9999 \
--datadir $DATA_DIR \
--inte_model transformer_dit \
--xh_hidden_size 382 --K 382 \
--hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4.0 --mlp_dropout 0.0
```

### Train GFD + EdgeDiT on GeomDrugs:
```bash
python /projects/prjs1459/Thesis/global_framework_pretrain_egnn/main_geom_drugs.py \
--n_epochs 3000 \
--exp_name ${EXP_NAME} \
--n_stability_samples 500 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_steps 1000 \
--diffusion_noise_precision 1e-5 \
--diffusion_loss_type l2 \
--batch_size 64 \
--nf 256 \
--n_layers 4 \
--lr 1e-4 \
--normalize_factors [1,4,10] \
--test_epochs 1 \
--ema_decay 0.9999 \
--normalization_factor 1 \
--model egnn_dynamics \
--visualize_every_batch 100000 \
--inte_model transformer_dit \
--xh_hidden_size 184 --K 184 \
--hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4.0 --mlp_dropout 0.0
```



## LFD

Train LFD + EdgeDiT with frame alignment loss on QM9:

```bash
python main_qm9.py \
--n_epochs 4350 \
--exp_name ${EXP_NAME} \
--n_stability_samples 1000 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 \
--diffusion_steps 1000 \
--diffusion_loss_type l2 \
--batch_size 256 \
--nf 256 \
--n_layers 9 \
--lr 2e-4 \
--normalize_factors [1,4,10] \
--test_epochs 20 \
--ema_decay 0.9999 \
--datadir $TMPDIR \
--inte_model transformer_dit \
--lambda_frame 0.1
```

## IFD
Train IFD + EdgeDiT on QM9:
```bash
python main_qm9.py \
--n_epochs 5000 \
--exp_name ${EXP_NAME} \
--n_stability_samples 1000 \
--diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 \
--diffusion_steps 1000 \
--diffusion_loss_type l2 \
--batch_size 256 \
--nf 256 \
--n_layers 9 \
--lr 2e-4 \
--normalize_factors [1,4,10] \
--test_epochs 20 \
--ema_decay 0.9999 \
--datadir $TMPDIR
```



## After training

To analyze the sample quality of molecules

```python eval_analyze.py --model_path outputs --n_samples 10_000```

To visualize some molecules

```python eval_sample.py --model_path outputs --n_samples 10_000```


