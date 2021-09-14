from mlp_gcn_train_test import train_test
import argparse

if __name__ == "__main__":
    dataset_path = "../Data/"  
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BRCA')
    args = parser.parse_args()
    data_folder = dataset_path + args.dataset
    view_list = [1,2,3]
    num_epoch_pretrain = 0
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    if data_folder == dataset_path + 'ROSMAP' or  data_folder == dataset_path + "LuadLusc100":
        num_class = 2
    if data_folder == dataset_path + 'BRCA' or data_folder == dataset_path + '5000samples':
        num_class = 5
    
    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch)
