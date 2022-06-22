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



def train_model(temperature,
                train_file_pattern,
                test_file_pattern,
                max_files_to_load=None,
                n_epochs=2000,
                n_particles=4096, 
		tcl=7,
                learning_rate=1e-4,
                grad_clip=1.0,
                measurement_store_interval=5,
                seed=0,
                if_learn_edge=False
                ):


    torch.manual_seed(seed) 
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainset = Dataset(if_learn_edge,
                       train_file_pattern,
                       max_files_to_load,
                       n_particles,
                       edge_threshold=2.0
                       )
                       
    testset = Dataset(if_learn_edge,
                      test_file_pattern,
                      max_files_to_load,
                      n_particles, 
                      edge_threshold=2.0,                      
                      if_train=False)

    train_batch_size = 5 if (if_learn_edge == False) else 1
    train_data_loader = torch_geometric.loader.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    test_data_loader = torch_geometric.loader.DataLoader(testset, batch_size=1, shuffle=False)
    
    model = Model(if_learn_edge).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    

    for epoch in range(n_epochs+1):

        train_losses = []        
        for data in train_data_loader:        
            data = data.to(device) 
            optimizer.zero_grad()                        
            params = model(data)   # common btwn node / edge
            loss = loss_fn(params, data.y)                    
            loss.backward()

            if grad_clip != float('inf'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_losses.append( loss.item() )
            

        if epoch%10==0 and test_data_loader is not None:    

            valid_losses = []
            valid_stats = []

            for data in test_data_loader:

                data = data.to(device)                
                prediction = model(data)
                loss = loss_fn(prediction, data.y)
                loss_value = loss.item()            
                valid_losses.append(loss_value)

                prediction_map = torch.squeeze(prediction).to('cpu').detach().numpy().copy()
                target_map = torch.squeeze(data.y).to('cpu').detach().numpy().copy()                                
                mask = torch.squeeze(data.mask).to('cpu').detach().numpy().copy()
                valid_stats.append(np.corrcoef(prediction_map[mask == True],target_map[mask == True])[0, 1] )
                
            fm = "loss_T" + '{:.2f}'.format(temperature) + "_tc" + str(tcl) + "_" + str(seed+1) + ".dat"
            with open(fm, 'a') as f:
                f.write( str(epoch) + "," + str(np.mean(train_losses)) + "," + str( np.mean(valid_losses)) + "," + str(np.mean(valid_stats)) + "\n")

                
            
