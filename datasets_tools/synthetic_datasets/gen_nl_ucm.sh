PYTHON="python"
N=500
T=100
n_states=2
n_obs=2
dataset_type="NL_UCM_SSM"
script_name="datasets_tools/synthetic_datasets/gen_data.py"
output_path="data/nl_ucm_data/"


q2=0.001 # -30dB

# r2 in 0.001 0.01 0.1 1 10 100 1000, corresponding to q2 in -30dB, vdB in 0,10,20,30,40,50,60 in the paper.

for r2 in 0.001 0.01 0.1 1 10 100 1000
do
    ${PYTHON} ${script_name} \
    --n_states ${n_states} \
    --n_obs ${n_obs} \
    --num_samples $N \
    --sequence_length $T \
    --r2 $r2 \
    --q2 $q2 \
    --dataset_type ${dataset_type} \
    --output_path ${output_path} \
    --force
done
