PYTHON="python"
N=1000
T=100
n_states=3
n_obs=3
dataset_type="LorenzSSM"
script_name="datasets_tools/lorenz_datasets/gen_data.py"
output_path="data/lorenz_data/"


q2=0.0001



for r2 in 1 10 100 1000
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
