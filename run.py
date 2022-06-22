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


import train
import os

temperature_list = [0.44, 0.50, 0.56, 0.64]
temperature = temperature_list[0]

time_index = 7




def main():

    # set where the dataset is located
    directory_pattern= "./small_data/"   

    train_file_pattern = 'T' + '{:.2f}'.format(temperature) + '/train/*tc' +  '{:02d}'.format(time_index) + '*.npz'
    train_file_pattern = os.path.join(directory_pattern, train_file_pattern)
    test_file_pattern = 'T' + '{:.2f}'.format(temperature) + '/test/*tc' + '{:02d}'.format(time_index) + '*.npz'
    test_file_pattern = os.path.join(directory_pattern, test_file_pattern)
    
    train.train_model(
        temperature=temperature,
        train_file_pattern=train_file_pattern,
        test_file_pattern=test_file_pattern,
        n_epochs=2000, 
        max_files_to_load=400,  # "./small_data/" = 20 for train, 5 for test 
        tcl=time_index,
        seed=0,
        if_learn_edge=False)
            
if __name__ == '__main__':
    main()
    
