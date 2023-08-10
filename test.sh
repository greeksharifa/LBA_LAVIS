export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# vicuna 7b
# TRANSFORMERS_VERBOSITY=info

port_value_file="./port_value_file.txt"

port_value=$(<$port_value_file)
echo "old: $port_value"

port_value=$(($port_value + 1))
echo "new port_value: $port_value"

echo ${port_value} > ${port_value_file}

CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node=4 --master_port=$port_value test.py --cfg-path instructBLIP_FT_vicuna7b.yaml
