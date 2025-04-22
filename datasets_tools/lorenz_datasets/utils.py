#####################################################
# Creator: Anubhab Ghosh
# Feb 2023
#####################################################
import numpy as np
import torch
from torch import nn
import os
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from collections import deque
import pickle as pkl
import json


def dB_to_lin(x):
    return 10**(x / 10)


def lin_to_dB(x):
    assert x != 0, 'X is zero'
    return 10 * np.log10(x)


def partial_corrupt(x, p=0.7, bias=0.0):

    if x < 0:
        p *= -1
        bias *= 1
    #return np.random.uniform(x, x*(1+p)) + bias
    return x * (1 + p) + bias


def generate_normal(N, mean, Sigma2):

    # n = N(mean, std**2)
    n = np.random.multivariate_normal(mean=mean, cov=Sigma2, size=(N, ))
    return n


def count_params(model):
    """Counts two types of parameters:

    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)
    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters()
                                     if p.requires_grad == True)
    return total_num_params, total_num_trainable_params


def mse_loss(x, xhat):
    loss = nn.MSELoss(reduction='none')
    return loss(xhat, x)


def mse_loss_dB(x, xhat):
    noise_p = mse_loss(xhat, x).mean((1, 2))
    return 10 * torch.log10(noise_p).mean()


def nmse_loss(x, xhat):
    #loss = nn.MSELoss(reduction='mean')
    #noise_p = loss(xhat, x)
    #signal_p = loss(x, torch.zeros_like(x))
    return mse_loss_dB(xhat, x) - mse_loss_dB(x, torch.zeros_like(x))
    #return 10*torch.log10(noise_p / signal_p)


def nmse_loss_std(x, xhat):
    loss = nn.MSELoss(reduction='none')
    noise_p = loss(xhat, x)
    signal_p = loss(x, torch.zeros_like(x))
    return (10 * torch.log10(noise_p.mean(
        (1, 2))) - 10 * torch.log10(signal_p.mean((1, 2)))).std()


def mse_loss_dB_std(x, xhat):
    loss = nn.MSELoss(reduction='none')
    noise_p = loss(xhat, x).mean((1, 2))
    return (10 * torch.log10(noise_p)).std()


def get_mvnpdf(mean, cov):

    distr = MultivariateNormal(loc=mean, covariance_matrix=cov)
    return distr


def sample_from_pdf(distr, N_samples=100):

    samples = distr.sample((N_samples, ))
    return samples


def compute_log_prob_normal(X, mean, cov):

    Lambda_cov, U_cov = torch.linalg.eig(cov)
    logprob_normal_fn = lambda X : torch.real(- 0.5 * X.shape[1] * torch.log(torch.Tensor([2*torch.pi])) - \
        0.5 * torch.sum(torch.log(Lambda_cov)) - \
        torch.diag(0.5 * (U_cov.H @ (X.H - mean).type(torch.cfloat)).H \
            @ torch.diag(1 / (Lambda_cov + 1e-16)) \
            @ (U_cov.H @ (X.H - mean).type(torch.cfloat))))

    pX = logprob_normal_fn(X)

    return pX


def check_psd_cov(C):
    C = C.detach().cpu().numpy()
    return np.all(np.linalg.eigvals(C) >= 0)


def compute_inverse(X):

    U, S, Vh = torch.svd(X)
    return Vh @ (torch.diag(1 / S.reshape((-1, ))) @ U.T)


def create_diag(x):
    return torch.diag_embed(x)


class Series_Dataset(Dataset):

    def __init__(self, Z_XY_dict):

        self.data_dict = Z_XY_dict
        self.trajectory_lengths = Z_XY_dict['trajectory_lengths']

    def __len__(self):

        return len(self.data_dict['data'])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'inputs': np.expand_dims(self.data_dict['data'][idx][1], axis=0),
            'targets': np.expand_dims(self.data_dict['data'][idx][0], axis=0)
        }

        return sample


def obtain_tr_val_test_idx(dataset,
                           tr_to_test_split=0.9,
                           tr_to_val_split=0.83):

    num_training_plus_test_samples = len(dataset)
    print('Total number of samples: {}'.format(num_training_plus_test_samples))
    print('Training + val to test split: {}'.format(tr_to_test_split))
    print('Training to val split: {}'.format(tr_to_val_split))

    num_train_plus_val_samples = int(tr_to_test_split *
                                     num_training_plus_test_samples)
    num_test_samples = num_training_plus_test_samples - num_train_plus_val_samples
    num_train_samples = int(tr_to_val_split * num_train_plus_val_samples)
    num_val_samples = num_train_plus_val_samples - num_train_samples

    indices = torch.randperm(num_training_plus_test_samples).tolist()
    tr_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:num_train_samples +
                          num_val_samples]
    test_indices = indices[num_train_samples + num_val_samples:]

    return tr_indices, val_indices, test_indices


def my_collate_fn(batch):
    inputs = [item['inputs'] for item in batch]
    targets = [item['targets'] for item in batch]
    targets = torch.from_numpy(np.row_stack(targets).astype(np.float32))
    inputs = torch.from_numpy(np.row_stack(inputs).astype(np.float32))
    return (inputs, targets)


def get_dataloaders(dataset,
                    batch_size,
                    tr_indices,
                    val_indices,
                    test_indices=None,
                    val_batch_size=None,
                    te_batch_size=None):

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(tr_indices),
        num_workers=0,
        collate_fn=my_collate_fn)

    if val_batch_size is None:
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            num_workers=0,
            collate_fn=my_collate_fn)
    else:
        val_loader = DataLoader(
            dataset,
            batch_size=val_batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            num_workers=0,
            collate_fn=my_collate_fn)

    if te_batch_size is None:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_indices),
            num_workers=0,
            collate_fn=my_collate_fn)
    else:
        test_loader = DataLoader(
            dataset,
            batch_size=te_batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_indices),
            num_workers=0,
            collate_fn=my_collate_fn)

    return train_loader, val_loader, test_loader


def create_splits_file_name(dataset_filename, splits_filename):

    idx_dset_info = dataset_filename.rfind('m')
    idx_splitfilename = splits_filename.rfind('.pkl')
    splits_filename_modified = splits_filename[:idx_splitfilename] + '.pkl'  #+ "_" + dataset_filename[idx_dset_info:]
    return splits_filename_modified


def create_file_paths(params_combination_list, filepath, main_exp_name):

    list_of_logfile_paths = []
    # Creating the logfiles
    for params in params_combination_list:

        exp_folder_name = 'trajectories_M{}_P{}_N{}/'.format(
            params['num_trajectories'], params['num_realizations'],
            params['N_seq'])

        #print(os.path.join(log_filepath, main_exp_name, exp_folder_name))
        full_path_exp_folder = os.path.join(filepath, main_exp_name,
                                            exp_folder_name)
        list_of_logfile_paths.append(full_path_exp_folder)
        os.makedirs(full_path_exp_folder, exist_ok=True)

    return list_of_logfile_paths


def get_list_of_config_files(model_type,
                             options,
                             dataset_mode='pfixed',
                             params_combination_list=None,
                             main_exp_name=None):

    #logfile_path = "./log/estimate_theta_{}/".format(dataset_mode)
    #modelfile_path = "./models/"
    if main_exp_name is None:
        main_exp_name = '{}_L{}_H{}_multiple'.format(
            model_type, options[model_type]['n_layers'],
            options[model_type]['n_hidden'])
    else:
        pass

    base_config_dirname = os.path.dirname(
        './config/configurations_alltheta_pfixed.json')

    list_of_config_folder_paths = create_file_paths(
        params_combination_list=params_combination_list,
        filepath=base_config_dirname,
        main_exp_name=main_exp_name)

    #list_of_gs_jsonfile_paths = create_file_paths(params_combination_list=params_combination_list,
    #                                        filepath=modelfile_path,
    #                                        main_exp_name=main_exp_name)

    list_of_config_files = []

    for i, config_folder_path in enumerate(list_of_config_folder_paths):

        config_filename = 'configurations_alltheta_pfixed_gru_M{}_P{}_N{}.json'.format(
            params_combination_list[i]['num_trajectories'],
            params_combination_list[i]['num_realizations'],
            params_combination_list[i]['N_seq'])
        os.makedirs(config_folder_path, exist_ok=True)
        config_file_name_full = os.path.join(config_folder_path,
                                             config_filename)
        list_of_config_files.append(config_file_name_full)

    # Print out the model files
    #print("Config files to be created at:")
    #print(list_of_config_files)

    return list_of_config_files


class NDArrayEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_splits_file(splits_filename):

    with open(splits_filename, 'rb') as handle:
        splits = pkl.load(handle)
    return splits


def load_saved_dataset(filename):

    with open(filename, 'rb') as handle:
        Z_XY = pkl.load(handle)
    return Z_XY


def save_dataset(Z_XY, filename):
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Saving the dataset
    with open(filename, 'wb') as handle:
        pkl.dump(Z_XY, handle, protocol=pkl.HIGHEST_PROTOCOL)


def check_if_dir_or_file_exists(file_path, file_name=None):
    flag_dir = os.path.exists(file_path)
    if not file_name is None:
        flag_file = os.path.isfile(os.path.join(file_path, file_name))
    else:
        flag_file = None
    return flag_dir, flag_file


class ConvergenceMonitor(object):

    def __init__(self, tol=1e-2, max_epochs=3):

        self.tol = tol
        self.max_epochs = max_epochs
        self.convergence_flag = False
        self.epoch_arr = [
        ]  # Empty list to store iteration numbers to check for consecutive iterations
        self.epoch_count = 0  # Counts the number of consecutive iterations
        self.epoch_prev = 0  # Stores the value of the previous iteration index
        self.history = deque()

    def record(self, current_loss):

        if np.isnan(current_loss) == False:

            # In case current_loss is not a NaN, it will continue to monitor
            if len(self.history) < 2:
                self.history.append(current_loss)
            elif len(self.history) == 2:
                _ = self.history.popleft()
                self.history.append(current_loss)

        else:

            # Empty the queue in case a NaN loss is encountered during training
            for _ in range(len(self.history)):
                _ = self.history.pop()

    def check_convergence(self):

        if (abs(self.history[0]) > 0) and (abs(
            (self.history[0] - self.history[-1]) / self.history[0]) <
                                           self.tol):
            convergence_flag = True
        else:
            convergence_flag = False
        return convergence_flag

    def monitor(self, epoch):

        if len(self.history) == 2 and self.convergence_flag == False:

            convg_flag = self.check_convergence()

            #if convg_flag == True and self.epoch_prev == 0: # If convergence is satisfied in first condition itself
            #print("Iteration:{}".format(epoch))
            #    self.epoch_count += 1
            #    self.epoch_arr.append(epoch)
            #    if self.epoch_count == self.max_epochs:
            #        print("Exit and Convergence reached after {} iterations for relative change in loss below :{}".format(self.epoch_count, self.tol))
            #        self.convergence_flag = True

            #elif convg_flag == True and self.epoch_prev == epoch-1: # If convergence is satisfied
            #print("Iteration:{}".format(epoch))
            if convg_flag == True and self.epoch_prev == epoch - 1:  # If convergence is satisfied
                self.epoch_count += 1
                self.epoch_arr.append(epoch)
                if self.epoch_count == self.max_epochs:
                    print('Consecutive iterations are:{}'.format(
                        self.epoch_arr))
                    print(
                        'Exit and Convergence reached after {} iterations for relative change in NLL below :{}'
                        .format(self.epoch_count, self.tol))
                    self.convergence_flag = True

            else:
                #print("Consecutive criteria failed, Buffer Reset !!")
                #print("Buffer State:{} reset!!".format(self.epoch_arr)) # Display the buffer state till that time
                self.epoch_count = 0
                self.epoch_arr = []
                self.convergence_flag = False

            self.epoch_prev = epoch  # Set iter_prev as the previous iteration index

        else:
            pass

        return self.convergence_flag


class NDArrayEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
