export CUDA_VISIBLE_DEVICES=0,1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# vicuna 7b
TRANSFORMERS_VERBOSITY=info python -m torch.distributed.run --nproc_per_node=1 --master_port=52426 \
  train.py --cfg-path instructBLIP_FT_vicuna7b.yaml
