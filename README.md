# PyG_BOTAN

This code offers the [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) implementation
of BOnd TArgetting Network (BOTAN).   By BOTAN, intricate relaxation processes in glassy dynamics can be precisely predicted from static partocile configuration, by setting neighbor-pair separation with time as its target quantity of learning.  This code can be positioned functionally as a straightfoward extension of a previous [TensorFlow/JAX code](https://github.com/deepmind/deepmind-research/tree/master/glassy_dynamics) provided by Deepmind & Google Brain group, offering the same feature of predicting particle propensity as well. 

This is a beta version at present.   The primary version of PyG_BOTAN, in which the simultaneous learning of relative motions (edge feature) and particle self-motion (node feature) will be released at an appropriate moment where the review process of our [paper](https://arxiv.org/abs/2206.*****) goes on.
Please [e-mail us](mailto:shiba@cc.u-tokyo.ac.jp) if you want use the provisional version of the code or the full part of our dataset. 

## Dataset
A small dataset for run test is attached in the directory named ``small_data`` in this repo. 

Our full dataset will be made open when our [paper](https://arxiv.org/abs/2206.*****) is accepted for formal publication. 


## How to use 
- Install [PyTorch](https://pytorch.org) (>=1.11.0) and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (>=2.0.4)

- Edit ``run.py`` to specify ``temperature`` (0.44 by default) and ``time_index`` (7 by deafult).  
``time_index`` indicates the time point, see [Supplemental Information](***) of our [paper](https://arxiv.org/abs/2206.*****).   
``time_index=7`` corresponds to  the alpha-relaxation time.  

- Run the code  
```python3 run.py```


## Data format
We provide our dataset, namely, results of our isoconfigurational computation with 500 independent initial configurations for 3D Kob-Andersen Lennard-Jones liquid, in a periodic cube box with fixed length ($L=15.05658$).
This dataset has some features in common with that by V. Bpast et al. but with several chnages in the setting. Hence, the dataset if. 


The data is stored in a uncompressed ``.npz`` binary format. Each ``.npz`` file contains three list of arrays with keys indicated below:

- `types`  
the particle types (0 == type A and 1 == type B) of the equilibrated system.
- `initial_positions`  
the particle positions of the equilibrated system.
- `positions`:   
the positions of the particles for each of the trajectories at selected time point.  Note that the shape of this numpy array is (32,4096,3), which means that particle positions of 32 isoconfigurational ensembles are stored with 4096 particles in three spatial dimensions. 

All units are in the nondimensional Lennard-Jones units.  Note that the positions are stored in the absolute coordinate system, so they can be out of the simulation box if the particle goes beyond the periodic boundary. 

## Cite

If you use this code for your research, please cite as:

```
@article {botan},
article_type = {journal},
title = {Unraveling intricate processes of glassy dynamics from static structure by machine learning relative motion},
author = {Hayato Shiba, Masatoshi Hanai, Toyotaro Suzumura, and Takashi Shimokawabe},
}
```


## Confirmed execution check environment
Note that PyTorch Geometric requires Python>=3.7. 
```
for NVIDIA GPUs -- CUDA 11.3 or above
for AMD GPUs (AMD Instinct Series) -- ROCm 4.5.2 or above

# Install PyTorch and PyG (for CUDA 11.3, RHEL8) 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```
The version requirement list for envieroment verification. 
```
numpy==1.22.4
scipy==1.8.1
six==1.16.0
torch==1.11.0+cu113
torch-cluster==1.6.0
torch-geometric==2.0.4
torch-scatter==2.0.9
torch-sparse==0.6.13
torch-spline-conv==1.2.1
torchaudio==0.11.0+cu113
torchvision==0.12.0+cu113
typing_extensions==4.2.0
```

