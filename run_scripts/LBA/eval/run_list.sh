# dramaqa
CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run --nproc_per_node=3 --master_port=55557 evaluate.py --cfg-path lavis/projects/LBA/eval/dramaqa_zeroshot_xinstructblip_eval.yaml --options model.decomposer_name="self" model.decomposition="zero-shot"
