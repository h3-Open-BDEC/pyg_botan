import os
import train

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
        p_frac=0.4, 
        temperature=temperature,
        train_file_pattern=train_file_pattern,
        test_file_pattern=test_file_pattern,
        n_epochs=1000,
        max_files_to_load=400,  
        tcl=time_index,
        seed=0)
        
        
if __name__ == '__main__':
    main()
    
