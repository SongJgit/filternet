#!/bin/bash
# This script is used to run the generate_data.py file for creating training data.
# Creator: Anubhab Ghosh, Feb 2024.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python"

# The number of i.i.d. trajectories each of length T that constitute the training data
N=1000

# Length of each such training data trajectory, default it is set to T=1000
T=100

# Number of hidden states in the process, usually for Lorenz (a.k.a. Lorenz-63), Chen
# attractors, the number of hidden states is equal to 3, while for Lorenz-96, this value must be changed to
# n_states= 20 (currently hardcoded in this manner) but can be in general n_states >= 4
n_states=3

# Number of observations in the measurement system
n_obs=3

# dataset_type defines the type of dynamical system, the general terminology, e.g. for the Lorenz 63 system,
# the type is LorenzSSM, similarly for Chen attractor we have ChenSSM.
# For the Lorenz-96 model, we have Lorenz96SSM.
# For underdetermined scenario with random matrix subsampled measurements: LorenzSSMrn${n_obs}, ChenSSMrn${n_obs},
# Lorenz96SSMrn${n_obs} with deterministic matrix subsampled measurements: LorenzSSMn${n_obs}, ChenSSMn${n_obs},
# Lorenz96SSMn${n_obs}.
# For the linear system, we have LinearSSM (can handle both full-rank, deterministic downsampled case).
dataset_type="LorenzSSM"

# The name of the script for generating data with full path name
script_name="datasets_tools/lorenz_datasets/generate_data.py"

# Output path to store the data
output_path="data/lorenz_data/"

# Set the process noise level (in dB)
sigma_e2_dB=-10.0

# For different signal-to-measurement-noise ratio (SMNRs), run the data generation
for smnr_dB in -10.0 0.0 10.0 20.0 30.0 # different values separated by space
do
    ${PYTHON} ${script_name} \
    --n_states ${n_states} \
    --n_obs ${n_obs} \
    --num_samples $N \
    --sequence_length $T \
    --sigma_e2_dB $sigma_e2_dB \
    --smnr_dB $smnr_dB \
    --dataset_type ${dataset_type} \
    --output_path ${output_path} \
    --force
done
