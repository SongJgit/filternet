import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from pathlib import Path
#import matplotlib.pyplot as plt
from utils import save_dataset, obtain_tr_val_test_idx
from parameters_opt import get_parameters
from ssm_models import LinearSSM, LorenzSSM, Lorenz96SSM
import argparse
# from parse import parse
import os
#####################################################
# Creator: Anubhab Ghosh
# Feb 2023
#####################################################
# Importing the necessary libraries


def initialize_model(type_, parameters):
    """This function initializes a SSM model object with parameter values from
    the dictionary `parameters`.

    Naming conventions:

    - <model_name>SSM: Represents a SSM model with equal dimension of the state and the observation vector,
    <model_name> being either 'Linear' (Linear SSM), 'Lorenz' (Lorenz-63 / Lorenz attractor), 'Chen' (Chen attractor),
    'Lorenz96' (high-dimensional Lorenz-96 attractor). The measurement matrix is usually an identity matrix.

    - <model_name>SSMrn<x>: Represents a SSM model with unequal (generally smaller dim. of observation than state)
    of the state and the observation vector, <model_name> being either 'Lorenz' (Lorenz-63 / Lorenz attractor),
    'Chen' (Chen attractor), 'Lorenz96' (high-dimensional Lorenz-96 attractor). The measurement matrix is usually a
    full matrix with i.i.d. Gaussian entries, with smaller no. of rows than cols.

    - <model_name>SSMn<x>: Represents a SSM model with unequal (generally smaller dim. of observation than state)
    of the state and the observation vector, <model_name> being either 'Lorenz' (Lorenz-63 / Lorenz attractor),
    'Chen' (Chen attractor), 'Lorenz96' (high-dimensional Lorenz-96 attractor). The measurement matrix is usually a
    block matrix with a low-rank identity matrix and a null-matrix/vector.

    Args:
        type_ (str): The name of the model
        parameters (_type_): parameter dictionary

    Returns:
        model: Object of model-class that is returned
    """
    if type_ == 'LinearSSM':

        model = LinearSSM(n_states=parameters['n_states'],
                          n_obs=parameters['n_obs'],
                          mu_e=parameters['mu_e'],
                          mu_w=parameters['mu_w'],
                          gamma=parameters['gamma'],
                          beta=parameters['beta'])

    elif type_ == 'LorenzSSM':

        model = LorenzSSM(n_states=parameters['n_states'],
                          n_obs=parameters['n_obs'],
                          J=parameters['J'],
                          delta=parameters['delta'],
                          delta_d=parameters['delta_d'],
                          alpha=parameters['alpha'],
                          decimate=parameters['decimate'],
                          mu_e=parameters['mu_e'],
                          mu_w=parameters['mu_w'])

    elif type_ == 'ChenSSM':

        model = LorenzSSM(n_states=parameters['n_states'],
                          n_obs=parameters['n_obs'],
                          J=parameters['J'],
                          delta=parameters['delta'],
                          delta_d=parameters['delta_d'],
                          alpha=parameters['alpha'],
                          decimate=parameters['decimate'],
                          mu_e=parameters['mu_e'],
                          mu_w=parameters['mu_w'])

    elif type_ == 'Lorenz96SSM' or 'Lorenz96SSMn' in type_ or 'Lorenz96SSMrn' in type_:

        model = Lorenz96SSM(n_states=parameters['n_states'],
                            n_obs=parameters['n_obs'],
                            delta=parameters['delta'],
                            delta_d=parameters['delta_d'],
                            F_mu=parameters['F_mu'],
                            method=parameters['method'],
                            H=parameters['H'],
                            decimate=parameters['decimate'],
                            mu_w=parameters['mu_w'])

    elif type_ == 'LorenzSSMn2' or 'LorenzSSMn1' or 'LorenzSSMrn2' or 'LorenzSSMrn3':

        model = LorenzSSM(n_states=parameters['n_states'],
                          n_obs=parameters['n_obs'],
                          J=parameters['J'],
                          delta=parameters['delta'],
                          delta_d=parameters['delta_d'],
                          alpha=parameters['alpha'],
                          H=parameters['H'],
                          decimate=parameters['decimate'],
                          mu_e=parameters['mu_e'],
                          mu_w=parameters['mu_w'])

    return model


def generate_SSM_data(model, T, sigma_e2_dB, smnr_dB):
    """This function generates a single pair (state trajectory, measurement
    trajectory) for the given ssm_model where the state trajectory has been
    generated using process noise corresponding to `sigma_e2_dB` and the
    measurement trajectory has been generated using measurement noise
    corresponding to `smnr_dB`.

    Args:
        model (object): ssm model object
        T (int): length of measurement trajectory, state trajectory is of length T+1 as it has an initial state of the ssm model
        sigma_e2_dB (float): process noise in dB scale
        smnr_dB (float): measurement noise in dB scale

    Returns:
        X_arr: numpy array of size (T+1, model.n_states)
        Y_arr: numpy array of size (T, model.n_obs)
    """

    X_arr, Y_arr = model.generate_single_sequence(T=T,
                                                  sigma_e2_dB=sigma_e2_dB,
                                                  smnr_dB=smnr_dB)

    return X_arr, Y_arr


def generate_state_observation_pairs(type_,
                                     parameters,
                                     T=100,
                                     N_samples=1000,
                                     sigma_e2_dB=-10.0,
                                     smnr_dB=10.0):
    """This function generate several state trajectory and measurement
    trajectory pairs.

    Args:
        type_ (str): string to indicate the type of SSM model
        parameters (dict): dictionary containing parameters for various SSM models
        T (int, optional): Sequence length of trajectories, currently all trajectories are of the same length. Defaults to 100.
        N_samples (int, optional): Number of sample trajectories. Defaults to 1000.
        sigma_e2_dB (float, optional): Process noise in dB scale. Defaults to -10.0.
        smnr_dB (float, optional): Measurement noise in dB scale. Defaults to 10.0.f

    Returns:
        Z_XY: dictionary containing the ssm model object used for generating data, the generated data, lengths of trajectories, etc.
    """
    Z_XY = {}
    Z_XY['num_samples'] = N_samples
    Z_XY_data_lengths = []
    Z_XY_data = []

    ssm_model = initialize_model(type_, parameters)
    # Z_XY['ssm_model'] = ssm_model

    samples = np.arange(N_samples)
    tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(
        dataset=samples, tr_to_test_split=0.9, tr_to_val_split=0.833)

    for i in range(N_samples):

        Xi, Yi = generate_SSM_data(ssm_model, T, sigma_e2_dB, smnr_dB)
        Z_XY_data_lengths.append(T)
        Z_XY_data.append([Xi[1:], Yi])
    sigma_w2 = ssm_model.sigma_w2
    Z_XY_data_lengths = np.vstack(Z_XY_data_lengths)
    Z_XY['train'] = {}
    Z_XY['train']['trajectory_lengths'] = Z_XY_data_lengths[tr_indices]
    Z_XY['train']['data'] = [Z_XY_data[i] for i in tr_indices]
    Z_XY['val'] = {}
    Z_XY['val']['trajectory_lengths'] = Z_XY_data_lengths[val_indices]
    Z_XY['val']['data'] = [Z_XY_data[i] for i in val_indices]
    Z_XY['test'] = {}
    Z_XY['test']['trajectory_lengths'] = Z_XY_data_lengths[test_indices]
    Z_XY['test']['data'] = [Z_XY_data[i] for i in test_indices]

    Z_XY['Covar_R'] = sigma_w2
    # Z_XY["data"] = np.row_stack(Z_XY_data).astype(object)
    #Z_pM["data"] = Z_pM_data
    # Z_XY["trajectory_lengths"] = np.vstack(Z_XY_data_lengths)
    # Z_XY["data"] = Z_XY_data
    return Z_XY


def create_filename(T=100,
                    N_samples=1000,
                    m=3,
                    n=3,
                    dataset_basepath='./data/',
                    type_='LorenzSSM',
                    sigma_e2_dB=-10,
                    smnr_dB=10):
    """Create the dataset based on the dataset parameters, currently this name
    is partially hard-coded and should be the same in the correspodning parsing
    function for the `main`.py files.

    Args:
        T (int, optional): Sequence length of trajectories. Defaults to 100.
        N_samples (int, optional): Number of sample trajectories. Defaults to 1000.
        m (int, optional): Dimension of the state vector. Defaults to 3.
        n (int, optional): Dimension of the meas. vector. Defaults to 3.
        dataset_basepath (str, optional): Location to save the data. Defaults to "./data/".
        type_ (str, optional): Type of SSM model. Defaults to "LorenzSSM".
        sigma_e2_dB (float, optional): Process noise in dB. Defaults to -10.
        smnr_dB (float, optional): Meas. noise in dB. Defaults to 10.

    Returns:
        dataset_fullpath: string describing output filename with full / absolute path.
    """

    datafile = 'trajectories_m_{}_n_{}_{}_data_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB.pkl'.format(
        m, n, type_, int(T), int(N_samples), sigma_e2_dB, smnr_dB)
    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath


def create_and_save_dataset(T,
                            N_samples,
                            filename,
                            parameters,
                            type_='LinearSSM',
                            sigma_e2_dB=0.1,
                            smnr_dB=10):
    """Calls most the functions above to generate data and saves it at the
    desired location given by `filename`.

    Args:
        T (int, optional): Sequence length of trajectories. Defaults to 100.
        N_samples (int, optional): Number of sample trajectories. Defaults to 1000.
        filename (str): string describing output filename with full / absolute path.
        parameters (dict): dictionary containing parameters for various SSM models
        m (int, optional): Dimension of the state vector. Defaults to 3.
        n (int, optional): Dimension of the meas. vector. Defaults to 3.
        dataset_basepath (str, optional): Location to save the data. Defaults to "./data/".
        type_ (str, optional): Type of SSM model. Defaults to "LorenzSSM".
        sigma_e2_dB (float, optional): Process noise in dB. Defaults to -10.
        smnr_dB (float, optional): Meas. noise in dB. Defaults to 10.
    """
    Z_XY = generate_state_observation_pairs(type_=type_,
                                            parameters=parameters,
                                            T=T,
                                            N_samples=N_samples,
                                            sigma_e2_dB=sigma_e2_dB,
                                            smnr_dB=smnr_dB)

    parent = os.path.dirname(filename)
    suffix = os.path.basename(filename).split('.')[-1]
    basename = os.path.basename(filename).split(f'.{suffix}')[0]

    new_filename = os.path.join(
        parent, basename + '_CovR{:.2f}.'.format(Z_XY['Covar_R']) + suffix)
    save_dataset(Z_XY, filename=new_filename)
    return new_filename


if __name__ == '__main__':

    usage = 'Create datasets by simulating state space models \n'\
            'Example usage (square brackets indicate meaningful values): \
            python generate_data.py --n_states [3] --n_obs [3] --num_samples [1000] --sequence_length [100] \
            --sigma_e2_dB [-10.0] --smnr_dB 10.0 --dataset_type [LinearSSM/LorenzSSM/Lorenz96SSM] \
            --output_path [./data/synthetic_data/stored_data]\n'\
            'Creates the dataset at the location output_path'\

    parser = argparse.ArgumentParser(
        description=
        'Input arguments related to creating a dataset for training learning-based methods like DANSE'
    )

    parser.add_argument(
        '--n_states',
        help='denotes the number of states in the latent model',
        type=int,
        default=5)
    parser.add_argument('--n_obs',
                        help='denotes the number of observations',
                        type=int,
                        default=5)
    parser.add_argument(
        '--num_samples',
        help=
        'denotes the number of trajectories to be simulated for each realization',
        type=int,
        default=500)
    parser.add_argument('--sequence_length',
                        help='denotes the length of each trajectory',
                        type=int,
                        default=200)
    parser.add_argument('--sigma_e2_dB',
                        help='denotes the process noise variance in dB',
                        type=float,
                        default=-10.0)
    parser.add_argument('--smnr_dB',
                        help='denotes the signal-to-measurement noise in dB',
                        type=float,
                        default=20.0)
    parser.add_argument(
        '--dataset_type',
        help=
        'specify type of the SSM (LinearSSM / LorenzSSM / ChenSSM / Lorenz96SSM)',
        type=str,
        default=None)
    parser.add_argument('--output_path',
                        help='Enter full path to store the data file',
                        type=str,
                        default=None)
    parser.add_argument('--force',
                        help='force to generate',
                        action='store_true')

    args = parser.parse_args()

    n_states = args.n_states
    n_obs = args.n_obs
    T = args.sequence_length
    N_samples = args.num_samples
    type_ = args.dataset_type
    output_path = args.output_path
    sigma_e2_dB = args.sigma_e2_dB
    smnr_dB = args.smnr_dB

    # Create the full path for the datafile
    datafilename = create_filename(T=T,
                                   N_samples=N_samples,
                                   m=n_states,
                                   n=n_obs,
                                   dataset_basepath=output_path,
                                   type_=type_,
                                   sigma_e2_dB=sigma_e2_dB,
                                   smnr_dB=smnr_dB)
    ssm_parameters, _ = get_parameters(n_states=n_states, n_obs=n_obs)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # If the dataset hasn't been already created, create the dataset
    # if not os.path.isfile(datafilename) or args.force:
    if len(os.listdir(output_path)) == 0 or args.force:
        # print('Creating the data file: {}'.format(datafilename))
        datafilename = create_and_save_dataset(
            T=T,
            N_samples=N_samples,
            filename=datafilename,
            type_=type_,
            parameters=ssm_parameters[type_],
            sigma_e2_dB=sigma_e2_dB,
            smnr_dB=smnr_dB)

    else:
        print('Dataset {} is already present!'.format(datafilename))

    print(f'Dataset saved successfully, at {datafilename}')
