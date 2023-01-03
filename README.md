# PyG_BOTAN

This code offers [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (PyG) implementation
of BOnd TArgetting Network (BOTAN, 牡丹), proposed in our [paper](https://arxiv.org/abs/2206.14024). By BOTAN, intricate relaxation processes in glassy dynamics can be predicted with high precision from the static particle configuration, by setting neighbor-pair separation with time as its target quantity of learning. From a functional viewpoint, this code can be positioned as a straightfoward extension of the previous [TensorFlow/JAX code](https://github.com/deepmind/deepmind-research/tree/master/glassy_dynamics) provided by Deepmind & Google Brain group, offering a feature of predicting particle propensity as well. 

This major version of PyG_BOTAN offers simultaneous learning of relative motions (edge feature) and particle self-motion (node feature).   

## Dataset
A small dataset for run test is attached in the directory named ``small_data`` in this repo. 

The full dataset for training and tests in our [paper](https://arxiv.org/abs/2206.14024) is available: [public_dataset.tar.gz] (~58 GByte). 



## Pretrained models 

It is essential for the high predictive accuracy of BOTAN that model parameters that are pretrined by edge target features.  It also enables you to get the results with a small number of tranining epochs before overlearning takes place.  By default, the code uses the pretrained model which are saved in ``./initial_model``. 
 
## How to use 
- Install [PyTorch](https://pytorch.org) (>=1.11.0) and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (>=2.0.4)  
Please refer to respective websites for installation.  PyTorch Geometric requires the three following packages, ``torch-geometric, torch-sparse, torch-scatter``, which take a certain lapse of time (typically a few minutes) for installtion.  
Make sure these packages are installed on the same CUDA (or ROCm) version. 

- Edit ``run.py`` to specify ``temperature`` (0.44 by default),  ``time_index``, and ``p_frac``.
- ``time_index`` indicates the time point, see [paper](https://arxiv.org/abs/2206.14024).   
The default is ``time_index=7``, alpha-relaxation time. 
- ``p_frac`` is a hyperparameter ( $p$ in our [paper](https://arxiv.org/abs/2206.14024), in the range \[0,1\])  determining the weight of losses between nodes and edges.  As the two extremes, the model  learns only the particle propensity (node target feature) when ``p_frac=1``,  and conversely it learns only the relative motion (edge target feature) ``p_frac=0``.  Set at 0.4 by default. 
- Run the code  
```python3 run.py```


## Data format
Our dataset contains results of isoconfigurational ensembles of simulation results for 3D Kob-Andersen Lennard-Jones liquid, generated from 500 independent initial configurations. Each isoconfigurational ensemble consists of 32 separate molecular trajectories generated with different initial velocities. Although this dataset has some features in common with that by V. Bapst et al, our dataset is independently generated by keeping the box length constant  at $L=15.05658$ all through and for all the temperatures. 

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
@misc{botan,
      title={Predicting the entire glassy dynamics from static structure by machine learning relative motion}, 
      author={Hayato Shiba and Masatoshi Hanai and Toyotaro Suzumura and Takashi Shimokawabe},
      year={2022},
      eprint={2206.14024},
      archivePrefix={arXiv},
      primaryClass={cond-mat.dis-nn}
}
```

## Confirmed execution check environment
Note that PyTorch Geometric requires Python>=3.7. 
```
for NVIDIA GPUs -- CUDA 11.3 or above
for AMD GPUs (AMD Instinct Series) -- ROCm 4.5.2 or above

# Install PyTorch and PyTorch Geometric (for CUDA 11.3, RHEL8) 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```
The version requirement list for environment verification. 
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

## Note on reproducibility of the results in our paper

Please note that the results for ``p_frac=1`` (node-targeting GNN) in our [paper](https://arxiv.org/abs/2206.14024) is produced by using an earlier version of this repository (v0.1). While this updated version would give similar results for  ``p_frac=1``, please note that full reproducibility is not guaranteed. 

## Acknowledgement
PyG_BOTAN is developed under the support of:
- [h3-open-BDEC project](https://h3-open-bdec.cc.u-tokyo.ac.jp) (JSPS KAKENHI Grant Number JP19H05662）
- "Joint Usage/Research Center for Interdisciplinary Large-scale Information Infrastructures" in Japan (Project ID: jh220052-NAH).  
- Code check & test by Dr. Masatoshi Kawai is acknowledged. 
