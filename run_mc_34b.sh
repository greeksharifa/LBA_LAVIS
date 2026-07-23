dataset_name=$1
device=$2
num_data=$3
exp=$4

CUDA_VISIBLE_DEVICES=2,3,4,5 python eval_llava_multiplechoice.py --path_qa_pair_csv ./data/multiple_choice_qa/${dataset_name}_sub_qas_val_xl_fewshot_vqaintrospect_unique.csv --path_video ./data/${dataset_name}/videos/%s.mp4 --path_result ./result_${dataset_name}/${exp} --llm_size 34b  --dataset_name ${dataset_name} --num_data ${num_data} --sub_qa_index ${exp}
# model_name = "llava-v1.6-vicuna-%s" % (llm_size)
# usage: bash run_mc.sh NExT_QA 4 10 base
# usage: bash run_mc.sh NExT_QA 4 10 1
