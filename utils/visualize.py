import json
import os
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

from utils.colors import Colors


def get_conf_rank(results, key, H):
    reverse_key = 'confidence_lba' if key == 'confidence_base' else 'confidence_base'
    results.sort(key=lambda x: x[reverse_key])
    N = len(results)
    for i, result in enumerate(results):
        result[f'rank_{reverse_key.split("_")[-1]}'] = int(i / N * H)
    results.sort(key=lambda x: x[key])
    for i, result in enumerate(results):
        result[f'rank_{key.split("_")[-1]}'] = int(i / N * H)
        
    return results


# def condition(conf_base, conf_lba, select_high_confidence, conf_gap):
    

def visualize(results, dataset, cfg, output_dir, total_base_match):
    if cfg.runner_cfg.visualize:
        output_dir = 'temp/'
    key = 'confidence_lba' if cfg.runner_cfg.get("threshold_lba", False) else 'confidence_base'
    
    N = len(results)
    M = max(1, N // cfg.runner_cfg.num_bin)
    H = cfg.runner_cfg.get("num_heatmap_row", 10)
    HH = max(1, N // H)
    
    results = get_conf_rank(results, key, H)
    # conf_gap = cfg.runner_cfg.get("conf_gap", 0.1)
    
    max_match, cur_match, min_match = total_base_match, total_base_match, total_base_match
    match_list = [cur_match]
    max_arg_confidence = -1e10
    confidence_percentile = 0.
    acc_base_list, acc_lba_list = [], []
    bins_base = [[] for _ in range(N // M + 1)]
    heatmap_data = {
        'base': [[[] for _ in range(N // HH + 1)] for _ in range(H)],
        'lba' : [[[] for _ in range(N // HH + 1)] for _ in range(H)],
    }
    heatmap_data2 = {
        'number': [[0 for _ in range(H)] for _ in range(H)],
        'change': [[0 for _ in range(H)] for _ in range(H)],
    }
    scatter_data = [] # pd.DataFrame(columns=['conf_base', 'conf_lba', 'acc_change'])
    
    max_conf_gap = 0.0
    for conf_gap in np.linspace(0, 1, 1001):
        cur_match = total_base_match
        
        for i, result in enumerate(results):
            acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
            acc_lba = dataset.get_accuracy(result['text_output_lba'], result['gt_ans'])
            
            if cfg.runner_cfg.select_high_confidence and result['confidence_base'] + conf_gap > result['confidence_lba']: # 높은것만 선택
                pass
            else: # 무조건 lba 선택
                cur_match += acc_lba - acc_base
            
            if cur_match > max_match:
                max_match = cur_match
                max_conf_gap = conf_gap
        
        print(f'\rconf_gap: {conf_gap:.3f} max_conf_gap: {max_conf_gap:.3f}, acc_base: {total_base_match / N * 100:.2f}, max_acc: {max_match / N * 100:.2f}', end='')
    
    print()
    max_match, cur_match, min_match = total_base_match, total_base_match, total_base_match
    
    for i, result in enumerate(results):
        acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
        acc_lba = dataset.get_accuracy(result['text_output_lba'], result['gt_ans'])
        acc_base_list.append(acc_base)
        acc_lba_list.append(acc_lba)
        
        bin_key = i // M
        bins_base[bin_key].append(acc_base)
        
        if cfg.runner_cfg.select_high_confidence and result['confidence_base'] + max_conf_gap > result['confidence_lba']: # 높은것만 선택
            pass
        else: # 무조건 lba 선택
            cur_match += acc_lba - acc_base
        
        match_list.append(cur_match)
        min_match = min(min_match, cur_match)
        
        if cur_match > max_match:
            max_match = cur_match
            max_arg_confidence = result[key]
            confidence_percentile = (i+1) / N * 100
            
        # for heatmap
        row = int(result[key] * H)
        col = i // HH
        heatmap_data['base'][row][col].append(acc_base)
        heatmap_data['lba'][row][col].append(acc_lba)
        
        if not cfg.runner_cfg.select_high_confidence or result['confidence_base'] + max_conf_gap < result['confidence_lba']:
        # if not cfg.runner_cfg.select_high_confidence or result['rank_base'] < result['rank_lba']:
            heatmap_data2['number'][H-1-result['rank_lba']][result['rank_base']] += 1
            heatmap_data2['change'][H-1-result['rank_lba']][result['rank_base']] += acc_lba - acc_base
            
        # for scatter plot
        scatter_data.append({
            'conf_base': np.log(result['confidence_base']), 
            'conf_lba': np.log(result['confidence_lba']),
            'acc_change': acc_lba - acc_base, 
            'size': abs(acc_lba - acc_base) * 2 + 3,
            'acc_base': acc_base,
            'acc_lba': acc_lba,
            'class': round(acc_base + acc_lba * 2),
        })
        
    final_acc_list = [match / N for match in match_list]
    
    metrics = OrderedDict()
    
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
        for q in ["TCDISPFL"]:
            for q_type in match_per_type.keys():
                if q_type.startswith(q) and total_per_type[q_type] > 0:
                    qtype_v = f'{match_per_type[q_type] / total_per_type[q_type] * 100:4.2f}% = {match_per_type[q_type]:6.1f} / {total_per_type[q_type]:5d}'
                    # print(f'{q_type:<21s}: {qtype_v}')
                    metrics[f'{q_type:<21s}'] = qtype_v
    
    
    # E_CR, E_IC: Error Correction raio / Error Induction ratio
    e_cr, e_ic = dataset.get_e_cr_e_ic(acc_base_list, acc_lba_list)
    
    results.sort(key=lambda x: x['confidence_lba'])
    bins_lba = [[] for _ in range(N // M + 1)]
    
    for i, result in enumerate(results):
        # bins_lba
        acc_lba = dataset.get_accuracy(result['text_output_lba'], result['gt_ans'])
        
        bin_key = i // M
        bins_lba[bin_key].append(acc_lba)
        
    
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    plt.plot([i / N * 100 for i, _ in enumerate(final_acc_list)], final_acc_list, color='b')
    plt.title(f'{cfg.datasets_cfg.dataset_name} | E_CR: {e_cr:.2f}%, E_IC: {e_ic:.2f}%')
    plt.xlabel('Confidence Percentile')
    plt.ylabel('Accuracy')
    plt.xticks(list(range(0, 101, 10)))# [0, 25, 50, 75, 100])
    
    plt.subplot(2, 1, 2)
    acc_bin_base = [sum(bin) / len(bin) for bin in bins_base if len(bin) > 0]
    acc_bin_lba = [sum(bin) / len(bin) for bin in bins_lba if len(bin) > 0]
    plt.plot([i for i in range(len(acc_bin_base))], acc_bin_base, color='r')
    plt.plot([i for i in range(len(acc_bin_lba))], acc_bin_lba, color='g')
    
    plt.title(f'{cfg.datasets_cfg.dataset_name} | acc for {cfg.runner_cfg.num_bin} bins by conf') # len(acc_bin)
    plt.xlabel('Confidence Percentile')
    plt.ylabel('Accuracy (Base: red, LBA: green)')
    plt.xticks([(cfg.runner_cfg.num_bin // 10) * i for i in range(11)], list(range(0, 101, 10)))
    
    # plt.xticks([(cfg.runner_cfg.num_bin // 5) * i for i in range(6)])
    plt.subplots_adjust(left=0.125, bottom=0.10, right=0.9, top=0.90, wspace=0.2, hspace=0.45)
    fig_path = os.path.join(output_dir, "acc_bin.png")
    # if not cfg.runner_cfg.visualize:
    plt.savefig(fig_path, dpi=300)
    print(f'saved fig path is {fig_path}')
    # draw heatmap
    # draw_heatmap(heatmap_data['base'], output_dir, 'base', N // H + 1)
    # draw_heatmap(heatmap_data['lba'], output_dir, 'lba', N // H + 1)
    draw_heatmap2(heatmap_data2['number'], output_dir, 'number', H)
    draw_heatmap2(heatmap_data2['change'], output_dir, 'change', H)
    
    # draw scatter plot
    plt.figure(figsize=(6,6))
    scatter_df = pd.DataFrame(scatter_data)#, columns=['conf_base', 'conf_lba', 'acc_change', 'size'])
    sns.scatterplot(x='conf_base', y='conf_lba', hue='class', data=scatter_df, palette='rainbow', # 'coolwarm'
                    hue_order=[1, 0, -1], size='size', sizes=(4, 10))
    plt.title(f'{cfg.datasets_cfg.dataset_name} | class 0: both wrong, 1: right -> wrong, 2: wrong -> right, 3: both right')
    plt.xlabel('Confidence Base')
    plt.ylabel('Confidence LBA')
    fig_path = os.path.join(output_dir, "scatter.png")
    plt.savefig(fig_path, dpi=300)
    print(f'saved scatter fig path is {fig_path}')
    
    print(f'saved config path is {os.path.join(output_dir, "config.yaml")}')

    
    metrics.update({
        "acc_origin           ": f'{total_base_match / N * 100:.2f}%',
        "max_acc_by_tau       ": f'{max(final_acc_list) * 100:.2f}%', 
        "max_arg_confidence   ": f'{max_arg_confidence:.6f}',
        "confidence_percentile": f'{confidence_percentile:.2f}%',
        "E_CR                 ": f'{e_cr:.2f}%',
        "E_IC                 ": f'{e_ic:.2f}%',
        "min_match            ": f'{min_match / N * 100:.2f}%',
    })
    print("metrics:", json.dumps(metrics, indent=4))

    # if not cfg.runner_cfg.visualize:
    with open(os.path.join(output_dir, "evaluate.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=4) + "\n")
            
    print(cfg.datasets_cfg.dataset_name, end='\t')
    print(cfg.runner_cfg.recomposer_name, end='\t')
    
    print(cfg.runner_cfg.get("select_high_confidence", False), end='\t')
    print(cfg.runner_cfg.get("train_recomposer_examplar", False), end='\t')
    # print(cfg.runner_cfg.get("threshold_lba", False), end='\t')
    print(cfg.runner_cfg.get("vision_supple", False), end='\t')
    print(cfg.runner_cfg.get("num_sub_qa_generate", 1), end='\t')
    # print(cfg.runner_cfg.get("conf_gap", 0.0), end='\t')
    print(f'{max_conf_gap:.3f}', end='\t')


    print(f'copy and paste: {total_base_match / N * 100:.2f}\t{max(final_acc_list) * 100:.2f}\t{max_arg_confidence:.6f}\t{confidence_percentile:.2f}\t{e_cr:.2f}\t{e_ic:.2f}', end='')
    if 'type' in results[0]:
        for q in ["TCDISPFL"]:
            for q_type in match_per_type.keys():
                if q_type.startswith(q) and total_per_type[q_type] > 0:
                    print(f'\t{match_per_type[q_type] / total_per_type[q_type] * 100:.2f}', end='')
    print('\n')


def draw_heatmap(raw_data, output_dir, key, col_num):
    data = [[-0.2 for _ in range(col_num)] for _ in range(10)]
    for i, row in enumerate(raw_data):
        for j, col in enumerate(row):
            if len(col) > 0:
                data[i][j] = sum(col) / len(col)

    # Convert the nested list to a numpy array
    data_array = np.array(data)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_array, annot=True, cmap='viridis', fmt='.2f')

    # Set title
    plt.xlabel('Relative Confidence')
    plt.ylabel('Absolute Confidence')
    plt.title(f'Acc_{key} Heatmap by Rel/Abs Confidence')

    # Show the plot
    # plt.show()
    plt.savefig(os.path.join(output_dir, f'heatmap_{key}.png'), dpi=300)


def draw_heatmap2(data, output_dir, key, H):
    # Convert the nested list to a numpy array
    data_array = np.array(data, dtype=int)
    
    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    
    if key == 'number':
        sns.heatmap(data_array, annot=True, cmap='viridis', fmt='d') #.1f
    else:
        sel_col = ['#ba001e','#d80019','#f32b1d','#ff502b','#ff7c3c','#ffa84e','#ffcb6c','#ffe992','#fcfeb3','#e4f693','#c6ea74','#a0de5c','#68cb57','#39be56','#00b14d','#00893e','#006b31']
        sel_colmap = ListedColormap(sel_col)
        
        sel_norm = list(range(-8, 9))
        sel_norm = BoundaryNorm(sel_norm, ncolors=len(sel_col))

        sns.heatmap(data_array, annot=True, cmap=sel_colmap, norm=sel_norm, fmt='+d')

    # Set title
    plt.xlabel('Base Confidence(rel %)')
    plt.ylabel('LBA Confidence(rel %)')
    plt.xticks(range(0, H+1), [round(100*i/H) for i in range(H+1)])
    plt.yticks(range(0, H+1), [round(100/H*(H-i)) for i in range(H+1)])
    plt.title(f'Base -> LBA {key}')

    # Show the plot
    # plt.show()
    fig_path = os.path.join(output_dir, f'heatmap_conversion_{key}.png')
    # fig_path = os.path.join('temp', f'heatmap_conversion_{key}.png')
    plt.savefig(fig_path, dpi=300)
    print(f'saved heatmap_conversion path is {fig_path}')


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
    
    return round(float(b < l)), round(float(b > l)) # w2r, r2w
    