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

import torch
import torch_geometric
from torch import Tensor
from torch_scatter import scatter
from typing import Optional, Tuple

class Model(torch.nn.Module):

    def __init__(self,
                 if_learn_edge=False):

        super().__init__()

        n_node_feature=1
        n_edge_feature=3
        mlp_size=64

        self.if_learn_edge = if_learn_edge

        self.node_encoder = MLP(n_node_feature, mlp_size)
        self.edge_encoder = MLP(n_edge_feature, mlp_size)

        self.decoder = MLP(mlp_size, mlp_size, decoder_layer=True)
        
        self.num_message_passing_steps = 7     
        self.network = Iterative_Layers( edge_model=EdgeUpdate(mlp_size),
                                         node_model=NodeUpdate(mlp_size),
                                         steps=self.num_message_passing_steps)        
        self.reset_parameters()        

    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder, self.network, self.decoder]: 
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

        
    def forward(self, data):   

        edge_index = data.edge_index    
        batch = data.batch

        encoded_x = self.node_encoder(data.x)
        encoded_edge_attr = self.edge_encoder(data.edge_attr)

        x, edge_attr = self.network(encoded_x,
                                    edge_index,
                                    encoded_edge_attr,
                                    batch)
                
        return self.decoder(x) if (self.if_learn_edge==False) else self.decoder(edge_attr)
        
        
    
class EdgeUpdate(torch.nn.Module):

    def __init__(self, mlp_size):
        super().__init__()
        self.mlp = MLP(4*mlp_size, mlp_size)

    def reset_parameters(self):
        self.mlp.reset_parameters()        
        
    def forward(self, src, dest, edge_attr, encoded_edge, batch=None):        
        edge_input = torch.cat([src, dest, edge_attr, encoded_edge], dim=1)   
        out = self.mlp(edge_input)
        return out


class NodeUpdate(torch.nn.Module):

    def __init__(self, mlp_size):
        
        super().__init__()
        self.mlp = MLP(3*mlp_size, mlp_size)

    def reset_parameters(self):
        self.mlp.reset_parameters()                

    def forward(self, x, edge_index, edge_attr, encoded_x, batch):
        row, col = edge_index
        recv = scatter( edge_attr, col, dim=0, dim_size=x.size(0) )
        out = torch.cat([x, encoded_x, recv], dim=1) 
        out = self.mlp(out)

        return out


class MLP(torch.nn.Module):

    def __init__(self, initial_size, mlp_size, decoder_layer=False):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(initial_size, mlp_size)
        self.fc2 = torch.nn.Linear(mlp_size, mlp_size)
        self.decoder_layer = decoder_layer
        if decoder_layer == True:
            self.fc3 = torch.nn.Linear(mlp_size, 1)


    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.decoder_layer == True:
            self.fc3.reset_parameters()
            

    def forward(self, x):

        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        if self.decoder_layer == True:
            x = self.fc3(x)

        return x


class Iterative_Layers(torch.nn.Module):

    def __init__(self, edge_model, node_model, steps):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.num_message_passing_steps = steps        
        self.reset_parameters()
        
    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Optional[Tensor] = None)  -> Tuple[Tensor, Tensor]:

        row = edge_index[0]   #src
        col = edge_index[1]   #dest
        encoded_x = x
        encoded_edge = edge_attr

        for _ in range(self.num_message_passing_steps):

            if self.edge_model is not None:

                edge_attr = self.edge_model(x[row],
                                            x[col],
                                            edge_attr,
                                            encoded_edge,
                                            batch)

                
            if self.node_model is not None:

                x = self.node_model(x,
                                    edge_index,
                                    edge_attr,
                                    encoded_x,
                                    batch)

        return x, edge_attr


