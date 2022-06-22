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

import glob
import re
import numpy as np
import torch
import torch_geometric


class Dataset(torch.nn.Module):

    def __init__(self,
                 if_learn_edge,
                 file_pattern,
                 max_files_to_load=None,
                 n_particles=4096, 
                 edge_threshold=2.0,                 
                 if_train=True
                 ):

        print("max_files_to_load = ", max_files_to_load)        

        filenames = sorted(glob.glob(file_pattern), key=self.natural_keys)        
        if max_files_to_load: filenames = filenames[:max_files_to_load]

        self.filenames = filenames    
        self.edge_threshold = edge_threshold
        self.num_nodes = n_particles        # number of particles: fixed
        self.phi = 1.2                      # number density of the system
        self.box = np.full(3, (float(self.num_nodes) / self.phi )**(1.0/3.0) ) # length of simulation box

        self.data = []
        self.init_positions_data = []

        for filename in self.filenames:
            print("loading: ", filename)

            npz = np.load(filename)  
            types = npz['types']
            initial_positions = npz['initial_positions']
            positions = npz['positions']
        
            node_feature, edge_index, edge_feature = self.make_graph_from_static_structure(initial_positions, types)

            target, mask = \
                self.get_targets_node(initial_positions, positions, types, if_train) if (if_learn_edge == False) \
                else self.get_targets_edge(initial_positions, positions,types, if_train)

            target = torch.tensor( target[:, None], dtype=torch.float )
            mask = torch.tensor( mask[:, None], dtype=torch.bool )

            graph = torch_geometric.data.Data(x=node_feature,
                                              edge_index=edge_index,
                                              edge_attr=edge_feature,
                                              y=target,
                                              mask=mask)

            self.init_positions_data.append(initial_positions)
            self.data.append(graph)
            


        
    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]
        
    def __len__(self):
        return len( self.filenames )


    def get_init_positions(self, idx):
        return self.init_positions_data[idx]
    
    
    def __getitem__(self, idx):
        graph = self.data[idx]
        graph = self.apply_random_rotation(graph)
        return graph


    def get_targets_node(self, initial_positions, trajectory_target_positions, types, if_train=True):
        """Returns the averaged particle mobilities from the sampled trajectories.  
        """    
        targets = np.mean([ np.linalg.norm(t - initial_positions, axis=-1)
                            for t in trajectory_target_positions], axis=0)

        mask = np.ones( len(trajectory_target_positions), dtype=np.bool )
        if ( if_train == False):
            mask = (types == 0).astype(np.bool)
        
        return targets.astype(np.float32), mask


    def get_targets_edge(self, initial_positions, trajectory_target_positions, types, if_train=True):

        cross_positions = initial_positions[None, :, :] - initial_positions[:, None, :]

        box_ = self.box[None, None, :]
        cross_positions += (cross_positions < -box_ / 2.).astype(np.float32) * box_
        cross_positions -= (cross_positions >= box_ / 2.).astype(np.float32) * box_

        distances = np.linalg.norm(cross_positions, axis=-1)
        indices = np.where( (distances < self.edge_threshold) & (distances > 1e-6) )

        edges = cross_positions[indices]
        edge_distances =  np.linalg.norm( edges, axis=1)

        cross_types = types[None, :]  + types[:, None]
        cross_types = cross_types[indices]
       
        mask = np.ones( len(edges), dtype=np.bool )
        if ( if_train == False):
            mask = (cross_types == 0).astype(np.bool) *  (edge_distances < 1.35).astype(np.bool)
       
        targets = [0.0] * len(edges)

        for t in trajectory_target_positions:
            t_cross_positions = t[ indices[1], :] - t[ indices[0], :]
            box__ = self.box[None, :]
            t_cross_positions += (t_cross_positions < -box__ / 2.).astype(np.float32) * box__
            t_cross_positions -= (t_cross_positions >= box__ / 2.).astype(np.float32) * box__
            t_distances = np.linalg.norm(t_cross_positions, axis=-1)
            
            t_edge_disp = t_distances - edge_distances 
            targets += t_edge_disp / len(trajectory_target_positions)

        
        return targets.astype(np.float32), mask

    
    def make_graph_from_static_structure(self, positions, types):

        cross_positions = positions[None, :, :] - positions[:, None, :]        
      
        box_ = self.box[None, None, :]
        cross_positions += (cross_positions < -box_ / 2.).astype(np.float32) * box_
        cross_positions -= (cross_positions > box_ / 2.).astype(np.float32) * box_
                                                                           
        distances = np.linalg.norm(cross_positions, axis=-1)
        indices = np.where( (distances < self.edge_threshold) & (distances > 1e-6) )

        node_feature = torch.tensor( types[:, None], dtype=torch.float )                
        edge_index = torch.tensor( np.array(indices), dtype=torch.long )        
        edge_feature = torch.tensor( cross_positions[indices], dtype=torch.float ) 
        
        return node_feature, edge_index, edge_feature



    def apply_random_rotation(self, graph):
        # Transposes edge features, so that the axes are in the first dimension.
        xyz = torch.transpose(graph.edge_attr, 0, 1)
        xyz = xyz[torch.randperm(3)]
        # Random reflections.
        symmetry =  np.random.randint(0, 2, [3])
        symmetry = torch.tensor( 1 - 2 * np.reshape(symmetry, [3, 1]), dtype=torch.float)
        xyz = xyz*symmetry 
        graph.edge_attr = torch.transpose(xyz, 0, 1)
    
        return graph
