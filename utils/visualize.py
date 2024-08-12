import json
import os
import matplotlib.pyplot as plt
from collections import Counter

from utils.colors import Colors


def visualize(results, dataset, cfg, output_dir, total_base_match):
    key = 'confidence_lba' if cfg.runner_cfg.threshold_lba else 'confidence_base'
    results.sort(key=lambda x: x[key])
    
    max_match, cur_match = total_base_match, total_base_match
    match_list = [cur_match]
    max_arg_confidence = -1e10
    confidence_percentile = 0.
    acc_base_list, acc_lba_list = [], []
    N = len(results)
    M = max(1, N // cfg.runner_cfg.num_bin)
    bins = [[] for _ in range(N // M + 1)]
    
    for i, result in enumerate(results):
        acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
        acc_lba = dataset.get_accuracy(result['text_output_lba'], result['gt_ans'])
        acc_base_list.append(acc_base)
        acc_lba_list.append(acc_lba)
        
        bin_key = i // M
        bins[bin_key].append(acc_base)
        
        if cfg.runner_cfg.select_high_confidence and result['confidence_base'] > result['confidence_lba']: # 높은것만 선택
            pass
        else: # 무조건 lba 선택
            cur_match += acc_lba - acc_base
        
        match_list.append(cur_match)
        
        if cur_match > max_match:
            max_match = cur_match
            max_arg_confidence = result['confidence_base']
            confidence_percentile = (i+1) / N * 100
            
    final_acc_list = [match / N for match in match_list]
    
    metrics = {}
    
    if 'type' in results[0]: # cfg.datasets_cfg.dataset_name in ['DramaQA', 'NExTQA', 'STAR']:
        match_per_type = {}    
        total_per_type = {}
        # total number of each question type
        # NExTQA : 4996 {'C': 2607, 'T': 1612, 'D': 777}
        # STAR   : 7098 {'Interaction': 2398, 'Sequence': 3586, 'Prediction': 624, 'Feasibility': 490}
        for i, result in enumerate(results):
            
            question_type = result['type']
            if cfg.datasets_cfg.dataset_name == 'NExTQA':
                question_type = question_type[0]
            
            target = result['gt_ans']
            
            # get predict
            if result['confidence_base'] <= max_arg_confidence:
                if cfg.runner_cfg.select_high_confidence:
                    if result['confidence_base'] > result['confidence_lba']:
                        predict = result['text_output_base']
                    else:
                        predict = result['text_output_lba']
                else:
                    predict = result['text_output_lba']
            else:
                predict = result['text_output_base']
            
            acc = dataset.get_accuracy(predict, target)
            if question_type not in match_per_type:
                match_per_type[question_type] = 0
                total_per_type[question_type] = 0
            else:
                match_per_type[question_type] += acc
                total_per_type[question_type] += 1
        
        # print("match_per_type:", match_per_type)
        for q_type in match_per_type.keys():
            if total_per_type[q_type] > 0:
                qtype_v = f'{match_per_type[q_type] / total_per_type[q_type] * 100:4.2f}% = {match_per_type[q_type]:6.1f} / {total_per_type[q_type]:5d}'
                # print(f'{q_type:<21s}: {qtype_v}')
                metrics[f'{q_type:<21s}'] = qtype_v
    
    
    # E_CR, E_IC: Error Correction raio / Error Induction ratio
    e_cr, e_ic = dataset.get_e_cr_e_ic(acc_base_list, acc_lba_list)
    
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1)
    # plt.subplots(constrained_layout=True)
    # tight_layout()
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    plt.plot([i / N * 100 for i, _ in enumerate(final_acc_list)], final_acc_list, color='b')
    plt.title(f'{dataset.__class__.__name__} | E_CR: {e_cr:.2f}%, E_IC: {e_ic:.2f}%')
    plt.xlabel('Confidence Percentile')
    plt.ylabel('Accuracy')
    plt.xticks([0, 25, 50, 75, 100])
    
    plt.subplot(2, 1, 2)
    acc_bin = [sum(bin) / len(bin) for bin in bins if len(bin) > 0]
    plt.plot([i for i in range(len(acc_bin))], acc_bin, color='r')
    plt.title(f'{dataset.__class__.__name__} | acc for {len(acc_bin)} bins')
    plt.xlabel('bins')
    plt.ylabel('Accuracy')
    plt.xticks([(cfg.runner_cfg.num_bin // 5) * i for i in range(6)])
    fig_path = os.path.join(output_dir, "acc_bin.png")
    # if not cfg.runner_cfg.visualize:
    plt.savefig(fig_path, dpi=300)
    print(f'saved fig path is {fig_path}')
    print(f'saved config path is {os.path.join(output_dir, "config.yaml")}')
    
    metrics.update({
        "acc_origin           ": f'{total_base_match / N * 100:.2f}%',
        "max_acc_by_tau       ": f'{max(final_acc_list) * 100:.2f}%', 
        "max_arg_confidence   ": f'{max_arg_confidence:.6f}',
        "confidence_percentile": f'{confidence_percentile:.2f}%',
        "E_CR                 ": f'{e_cr:.2f}%',
        "E_IC                 ": f'{e_ic:.2f}%',
    })
    print("metrics:", json.dumps(metrics, indent=4))

    # if not cfg.runner_cfg.visualize:
    with open(os.path.join(output_dir, "evaluate.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=4) + "\n")
            
    print(cfg.datasets_cfg.dataset_name, end='\t')
    print(cfg.runner_cfg.recomposer_name, end='\t')
    
    print(cfg.runner_cfg.select_high_confidence, end='\t')
    print(cfg.runner_cfg.train_recomposer_examplar, end='\t')
    print(cfg.runner_cfg.threshold_lba, end='\t')
    print(cfg.runner_cfg.random_frame, end='\t')

    print(cfg.runner_cfg.num_sub_qa_generate, end='\t')

    print(f'copy and paste: {total_base_match / N * 100:.2f}\t{max(final_acc_list) * 100:.2f}\t{max_arg_confidence:.6f}\t{confidence_percentile:.2f}\t{e_cr:.2f}\t{e_ic:.2f}', end='')
    if 'type' in results[0]:
        for q in ["TCDISPF"]:
            for q_type in match_per_type.keys():
                if q_type.startswith(q) and total_per_type[q_type] > 0:
                    print(f'\t{match_per_type[q_type] / total_per_type[q_type] * 100:.2f}', end='')
    print('\n')


def modefinder(l):   #numbers는 리스트나 튜플 형태의 데이터
    c = Counter(l)
    mode = c.most_common(1)
    return mode[0][0]

        
def sample_print(base, lba, gt_ans, get_accuracy, i):
    b, l = get_accuracy(base, gt_ans), get_accuracy(lba, gt_ans)
    
    if b < l:    # wrong -> right
        color = Colors.BRIGHT_GREEN
    elif b > l:  # right -> wrong
        color = Colors.BRIGHT_RED
    elif b == l and round(b) == 1:  # right -> right
        color = Colors.WHITE
    else:        # wrong -> wrong
        color = Colors.BRIGHT_YELLOW
        
    print(color, f'{base} -> {lba}, gt: {modefinder(gt_ans) if isinstance(gt_ans, list) else gt_ans}', Colors.RESET, end='\t')
    if i % 4 == 3:
        print()
    