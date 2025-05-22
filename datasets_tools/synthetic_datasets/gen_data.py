from typing import Tuple, Iterable
import os
import argparse
import pickle as pkl
from filternet.utils import logger
import numpy as np
from datetime import datetime
import torch
from datasets_tools.synthetic_datasets.params import get_parameters
from models import LorenzSSM, LinearSSM, Lorenz96SSM, UniformCircularMotionSSM


def obtain_tr_val_test_idx(dataset: Iterable,
                           tr_ratio: float = 0.8,
                           val_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This function returns the indices for the training, validation and test
    sets. The ratio of the training, validation and test sets is given by the
    ratios `tr_ratio`, `val_ratio` and `1-tr_ratio-val_ratio`

    Args:
        dataset (_type_):
        tr_ratio (float, optional): _description_. Defaults to 0.8.
        val_ratio (float, optional): _description_. Defaults to 0.1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
    """
    num_samples = len(dataset)
    logger.info('Total number of samples: {}'.format(num_samples))
    indices = torch.randperm(num_samples)

    train_size = int(tr_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    logger.info(f'Training size: {train_size}, Val size: {val_size}, Test size: {test_size}')

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[-test_size:]
    return train_indices, val_indices, test_indices


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
    print(type_)
    if type_ == 'LinearSSM':

        model = LinearSSM(n_states=parameters['n_states'],
                          n_obs=parameters['n_obs'],
                          mean_q=parameters['mean_q'],
                          mean_r=parameters['mean_r'],
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
                          mean_q=parameters['mean_q'],
                          mean_r=parameters['mean_r'])

    elif type_ == 'ChenSSM':

        model = LorenzSSM(n_states=parameters['n_states'],
                          n_obs=parameters['n_obs'],
                          J=parameters['J'],
                          delta=parameters['delta'],
                          delta_d=parameters['delta_d'],
                          alpha=parameters['alpha'],
                          decimate=parameters['decimate'],
                          mean_q=parameters['mean_q'],
                          mean_r=parameters['mean_r'])

    elif type_ in ['Lorenz96SSM', 'Lorenz96SSMn', 'Lorenz96SSMrn']:

        model = Lorenz96SSM(n_states=parameters['n_states'],
                            n_obs=parameters['n_obs'],
                            delta=parameters['delta'],
                            delta_d=parameters['delta_d'],
                            F_mu=parameters['F_mu'],
                            method=parameters['method'],
                            H=parameters['H'],
                            decimate=parameters['decimate'],
                            mean_r=parameters['mean_r'])

    elif type_ in ['LorenzSSMn2', 'LorenzSSMn1', 'LorenzSSMrn2', 'LorenzSSMrn3']:
        model = LorenzSSM(n_states=parameters['n_states'],
                          n_obs=parameters['n_obs'],
                          J=parameters['J'],
                          delta=parameters['delta'],
                          delta_d=parameters['delta_d'],
                          alpha=parameters['alpha'],
                          H=parameters['H'],
                          decimate=parameters['decimate'],
                          mean_q=parameters['mean_q'],
                          mean_r=parameters['mean_r'])

    elif type_ in ['NL_UCM_SSM', 'L_UCM_SSM']:
        model = UniformCircularMotionSSM(theta=parameters['theta'],
                                         linear_H=parameters['linear_H'],
                                         mean_q=parameters['mean_q'],
                                         mean_r=parameters['mean_r'])

    else:
        raise ValueError('Unknown model type')

    return model


def create_filename(T=100, N_samples=1000, m=3, n=3, dataset_basepath='./data/', type_='LorenzSSM', r2=0, q2=0):
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

    Returns:
        dataset_fullpath: string describing output filename with full / absolute path.
    """
    # format to "YYMMDDHHMM"
    now = datetime.now()
    formatted_time = str(now.strftime('%Y%m%d%H'))[2:]
    datafile = f'{type_}_{m}x{n}_T{T}_N{N_samples}_q2_{q2}_r2_{r2}.pkl'
    dataset_fullpath = os.path.join(dataset_basepath, formatted_time, datafile)
    if not os.path.exists(os.path.dirname(dataset_fullpath)):
        os.mkdir(os.path.dirname(dataset_fullpath))
    return dataset_fullpath


def generate_SSM_data(model, T, r2, q2):
    """This function generates a single pair (state trajectory, measurement
    trajectory) for the given ssm_model where the state trajectory has been
    generated using process noise corresponding to `sigma_e2_dB` and the
    measurement trajectory has been generated using measurement noise
    corresponding to `smnr_dB`.

    Args:
        model (object): ssm model object
        T (int): length of measurement trajectory, state trajectory is of length T+1 as it has an initial \
            state of the ssm model
        sigma_e2_dB (float): process noise in dB scale
        smnr_dB (float): measurement noise in dB scale

    Returns:
        X_arr: numpy array of size (T+1, model.n_states)
        Y_arr: numpy array of size (T, model.n_obs)
    """

    X_arr, Y_arr = model.generate_single_sequence(T=T, r2=r2, q2=q2)

    return X_arr, Y_arr


def generate_state_observation_pairs(type_, parameters, T=100, N_samples=1000, r2=None, q2=None):
    """This function generate several state trajectory and measurement
    trajectory pairs.

    Args:
        type_ (str): string to indicate the type of SSM model
        parameters (dict): dictionary containing parameters for various SSM models
        T (int, optional): Sequence length of trajectories, currently all trajectories are \
            of the same length. \Defaults to 100.
        N_samples (int, optional): Number of sample trajectories. Defaults to 1000.
        sigma_e2_dB (float, optional): Process noise in dB scale. Defaults to -10.0.
        smnr_dB (float, optional): Measurement noise in dB scale. Defaults to 10.0.f

    Returns:
        Z_XY: dictionary containing the ssm model object used for generating data, \
            the generated data, lengths of trajectories, etc.
    """
    Z_XY = {}
    Z_XY['num_samples'] = N_samples
    Z_XY_data_lengths = []
    Z_XY_data = []

    ssm_model = initialize_model(type_, parameters)
    # Z_XY['ssm_model'] = ssm_model

    samples = np.arange(N_samples)
    tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=samples)

    for i in range(N_samples):

        Xi, Yi = generate_SSM_data(ssm_model, T, r2, q2)
        Z_XY_data_lengths.append(T)
        Z_XY_data.append([Xi[1:], Yi])
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

    test_y = [
        torch.from_numpy(traj[0]).to(torch.float32).T
        for traj in Z_XY['test']['data']  # [(dim_state, traj length), ...]
    ]
    test_x = [
        torch.from_numpy(traj[1]).to(torch.float32).T for traj in Z_XY['test']['data']  # [(dim_obs, traj length),...]
    ]
    mse = torch.nn.MSELoss()
    rmse = mse(torch.cat(test_x), torch.cat(test_y)).sqrt()
    Z_XY['test']['obs_rmse'] = rmse.item()

    return Z_XY


def create_and_save_dataset(T, N_samples, filename, parameters, type_='LorenzSSM', r2=0.1, q2=0.1):
    """Calls most the functions above to generate data and saves it at the
    desired location given by `filename`.

    Args:
        T (_type_): _description_
        N_samples (_type_): _description_
        filename (_type_): _description_
        parameters (_type_): _description_
        type_ (str, optional): _description_. Defaults to 'LorenzSSM'.
        r2 (float, optional): _description_. Defaults to 0.1.
        q2 (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    Z_XY = generate_state_observation_pairs(type_=type_, parameters=parameters, T=T, N_samples=N_samples, r2=r2, q2=q2)

    parent = os.path.dirname(filename)
    suffix = os.path.basename(filename).split('.')[-1]
    basename = os.path.basename(filename).split(f'.{suffix}')[0]

    new_filename = os.path.join(parent, basename + '_TestObsRMSE{:.2f}.'.format(Z_XY['test']['obs_rmse']) + suffix)

    pkl.dump(Z_XY, open(new_filename, 'wb'))
    return new_filename


if __name__ == '__main__':

    usage = 'Create datasets by simulating state space models'
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--n_states', help='denotes the number of states in the latent model', type=int, default=5)
    parser.add_argument('--n_obs', help='denotes the number of observations', type=int, default=5)
    parser.add_argument('--num_samples',
                        help='denotes the number of trajectories to be simulated for each realization',
                        type=int,
                        default=500)
    parser.add_argument('--sequence_length', help='denotes the length of each trajectory', type=int, default=200)
    parser.add_argument('--r2', help='denotes the process noise variance', type=float, default=-10.0)
    parser.add_argument('--q2', help='denotes the process noise variance', type=float, default=-10.0)

    parser.add_argument(
        '--dataset_type',
        help='specify type of the SSM (LinearSSM / LorenzSSM / ChenSSM / Lorenz96SSM, NL_UCM_SSM, L_UCM_SSM)',
        type=str,
        default=None)
    parser.add_argument('--output_path', help='Enter full path to store the data file', type=str, default=None)
    parser.add_argument('--force', help='force to generate', action='store_true')

    args = parser.parse_args()

    n_states = args.n_states
    n_obs = args.n_obs
    T = args.sequence_length
    N_samples = args.num_samples
    type_ = args.dataset_type
    output_path = args.output_path
    r2 = args.r2
    q2 = args.q2
    logger.info(f'Generating {r2=}, {q2=}')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Create the full path for the datafile
    datafilename = create_filename(T=T,
                                   N_samples=N_samples,
                                   m=n_states,
                                   n=n_obs,
                                   dataset_basepath=output_path,
                                   type_=type_,
                                   r2=r2,
                                   q2=q2)
    ssm_parameters = get_parameters(n_states=n_states, n_obs=n_obs)

    if len(os.listdir(output_path)) == 0 or args.force:

        datafilename = create_and_save_dataset(T=T,
                                               N_samples=N_samples,
                                               filename=datafilename,
                                               type_=type_,
                                               parameters=ssm_parameters[type_],
                                               r2=r2,
                                               q2=q2)

    else:
        logger.info('Dataset {} is already present!'.format(datafilename))

    logger.info(f'Dataset saved successfully, at {datafilename}')
