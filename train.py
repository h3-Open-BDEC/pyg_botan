#
# Copyright 2022 Hayato Shiba. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import logging
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F 

from model import Model 
from load_data import Dataset


def train_model(p_frac,
                temperature,
                train_file_pattern,
                test_file_pattern,
                max_files_to_load=None,
                n_epochs=1000,
                n_particles=4096, 
		tcl=7,
                learning_rate=1e-4,
                grad_clip=1.0,
                measurement_store_interval=5,
                seed=0
                ):


    torch.manual_seed(seed) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainset = Dataset(train_file_pattern,
                       max_files_to_load,
                       n_particles,
                       edge_threshold=2.0
                       )
    
    testset = Dataset(test_file_pattern,
                      max_files_to_load,
                      n_particles, 
                      edge_threshold=2.0,                      
                      if_train=False)

    train_batch_size = 5 if (p_frac > 1.0 - 1e-6) else 1
    
    train_data_loader = torch_geometric.loader.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    test_data_loader = torch_geometric.loader.DataLoader(testset, batch_size=1, shuffle=False)
    
    model = Model().to(device)

    if (p_frac < 1.0 - 1e-6):
        mmp = "./initial_model/model" + str(seed+1) + ".pth"
        model.load_state_dict(torch.load(mmp))
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    
    for epoch in range(n_epochs+1):

        train_losses = []        
        for data in train_data_loader:        
            data = data.to(device) 
            optimizer.zero_grad()                        
            params_node, params_edge = model(data)   # common btwn node / edge
            loss_node = loss_fn(params_node, data.y)
            loss_edge = loss_fn(params_edge, data.y_edge)
            loss = p_frac * loss_node + (1.0 - p_frac) * loss_edge
            loss.backward()

            if grad_clip != float('inf'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_losses.append( loss.item() )
            

        if epoch%5==0 and test_data_loader is not None:    

            valid_losses = []
            valid_stats_node = []
            valid_stats_edge = []

            cnt = 0
            for data in test_data_loader:

                data = data.to(device)                
                prediction_node, prediction_edge = model(data)
                loss_node = loss_fn(prediction_node, data.y)
                loss_edge = loss_fn(prediction_edge, data.y_edge)
                loss = p_frac * loss_node + (1.0 - p_frac) * loss_edge                
                loss_value = loss.item()            
                valid_losses.append(loss_value)

                prediction_node = torch.squeeze(prediction_node).to('cpu').detach().numpy().copy()
                target_node = torch.squeeze(data.y).to('cpu').detach().numpy().copy()
                mask_node = torch.squeeze(data.mask).to('cpu').detach().numpy().copy()

                prediction_edge = torch.squeeze(prediction_edge).to('cpu').detach().numpy().copy()
                target_edge = torch.squeeze(data.y_edge).to('cpu').detach().numpy().copy()
                mask_edge = torch.squeeze(data.mask_edge).to('cpu').detach().numpy().copy()

                valid_stats_node.append(np.corrcoef(prediction_node[mask_node == True],target_node[mask_node == True])[0, 1] )
                valid_stats_edge.append(np.corrcoef(prediction_edge[mask_edge == True],target_edge[mask_edge == True])[0, 1] )

            node_score = np.mean(valid_stats_node) if (p_frac > 1e-6) else 0
            edge_score = np.mean(valid_stats_edge) if (p_frac < 1.0 - 1e-6) else 0

            fm = "./loss_p" + '{:.2f}'.format(p_frac)   + "_T" + '{:.2f}'.format(temperature) + "_tc" + str(tcl) + "_" + str(seed+1) + ".dat"
            with open(fm, 'a') as f:
                f.write( str(epoch) + "," + str(np.mean(train_losses)) + "," + str( np.mean(valid_losses)) + "," + str(node_score) + "," + str(edge_score) + "\n")

