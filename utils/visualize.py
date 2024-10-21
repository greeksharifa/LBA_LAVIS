import json
import os
import matplotlib.pyplot as plt
from pprint import pprint
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


def visualize(results, dataset, cfg, output_dir, total_base_match):
    saved_output_dir = output_dir
    if cfg.runner_cfg.visualize:
        output_dir = 'temp/'
    key = 'confidence_lba' if cfg.runner_cfg.get("threshold_lba", False) else 'confidence_base'
    
    N = len(results)
    M = max(1, N // cfg.runner_cfg.num_bin)
    H = cfg.runner_cfg.get("num_heatmap_row", 10)
    
    results = get_conf_rank(results, key, H)
    
    max_match, cur_match, min_match = total_base_match, total_base_match, total_base_match
    if type(cfg.runner_cfg.get("max_conf_gap", None)) == float:# is not None:
        max_conf_gap = cfg.runner_cfg.max_conf_gap
    else:
        max_conf_gap = 1e-100# 0.0
    
        if cfg.runner_cfg.select_high_confidence:
            conf_list = []
            conf_value = 1.0
            for _ in range(500):
                conf_list.append(conf_value)
                conf_value *= 0.98
            for _ in range(500):
                conf_list.append(conf_value)
                conf_value *= 0.6
            conf_list = conf_list[::-1]
            print(f'conf_list: {conf_list[:2]} ... {conf_list[-2:]}')
            # conf_list = list(np.linspace(0, 0.0001, 201))[:-1] + list(np.linspace(0.0001, 0.01, 199))[:-1] + list(np.linspace(0.01, 1, 991))
            for c_idx, conf_gap in enumerate(conf_list):
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
                
                tau2 = f'{conf_gap:.6g}'
                max_tau2 = f'{max_conf_gap:.6g}'
                print(f'\rc_idx: {c_idx:4d} | tau2: {tau2:15s} max_tau2: {max_tau2:15s} | cur_acc: {cur_match / N * 100:.2f} | acc_base: {total_base_match / N * 100:.2f}, max_acc: {max_match / N * 100:.2f}', end=' ' * 4)
    print()
        
    max_match, cur_match, min_match = total_base_match, total_base_match, total_base_match
    baseline_cur_match, baseline_max_match = total_base_match, total_base_match
    match_list, baseline_match_list = [cur_match], [cur_match]
    max_arg_confidence = -1e10
    confidence_percentile = 0.
    acc_base_list, acc_lba_list = [], []
    bins_base = [[] for _ in range(N // M + 1)]
    heatmap_data = {
        'number': [[0 for _ in range(H)] for _ in range(H)],
        'change': [[0 for _ in range(H)] for _ in range(H)],
        'change_abs': [[0 for _ in range(H)] for _ in range(H)],
    }
    scatter_data = [] # pd.DataFrame(columns=['conf_base', 'conf_lba', 'acc_change'])
    
    for i, result in enumerate(results):
        acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
        acc_base_kh = dataset.get_accuracy(result['text_outputs_lba_list'][0], result['gt_ans'])
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
            
        # baseline
        baseline_cur_match += acc_base_kh - acc_base # baseline_cur_match += acc_lba - acc_base
        baseline_match_list.append(baseline_cur_match)
        if baseline_cur_match > baseline_max_match:
            baseline_max_match = baseline_cur_match
            
        # heatmap
        if not cfg.runner_cfg.select_high_confidence or result['confidence_base'] + max_conf_gap < result['confidence_lba']:
        # if not cfg.runner_cfg.select_high_confidence or result['rank_base'] < result['rank_lba']:
            heatmap_data['number'][H-1-result['rank_lba']][result['rank_base']] += 1
            heatmap_data['change'][H-1-result['rank_lba']][result['rank_base']] += acc_lba - acc_base
            
        if result['confidence_lba'] != 0. and result['confidence_base'] != 0.:
            x_i = int(-2*(np.log(result['confidence_lba'])-1e-20))
            y_i = int(H+2*(np.log(result['confidence_base'])-1e-20))
            # print(result['confidence_lba'], result['confidence_base'], x_i, y_i)
            if 0 <= x_i < H and 0 <= y_i < H:
                heatmap_data['change_abs'][x_i][y_i] += acc_lba - acc_base
            
        # scatter plot
        if acc_base == 0 and acc_lba == 0:
            class_label = 'w2w'
        elif acc_base == 0 and acc_lba == 1:
            class_label = 'w2r'
        elif acc_base == 1 and acc_lba == 0:
            class_label = 'r2w'
        elif acc_base == 1 and acc_lba == 1:
            class_label = 'r2r'
        scatter_data.append({
            'conf_base': np.log(result['confidence_base']), 
            'conf_lba': np.log(result['confidence_lba']),
            'acc_change': acc_lba - acc_base, 
            'size': abs(acc_lba - acc_base) * 2 + 3,
            'acc_base': acc_base,
            'acc_lba': acc_lba,
            'class': class_label,
        })
        
    final_acc_list = [match / N for match in match_list]
    baseline_acc_list = [match / N for match in baseline_match_list]
    
    # E_CR, E_IC: Error Correction raio / Error Induction ratio
    e_cr, e_ic = dataset.get_e_cr_e_ic(acc_base_list, acc_lba_list)
    
    results.sort(key=lambda x: x['confidence_lba'])
    bins_lba = [[] for _ in range(N // M + 1)]
    
    for i, result in enumerate(results):
        # bins_lba
        acc_lba = dataset.get_accuracy(result['text_output_lba'], result['gt_ans'])
        
        bin_key = i // M
        bins_lba[bin_key].append(acc_lba)
        
    pprint(results[0], width=300)
    
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    plt.plot([i / N * 100 for i, _ in enumerate(final_acc_list)], final_acc_list, color='b')
    plt.plot([i / N * 100 for i, _ in enumerate(baseline_acc_list)], baseline_acc_list, color='gray')
    plt.title(f'{cfg.datasets_cfg.dataset_name} | E_CR: {e_cr:.2f}%, E_IC: {e_ic:.2f}%')
    plt.xlabel('Confidence Percentile')
    plt.ylabel('Accuracy')
    plt.xticks(list(range(0, 101, 10)))# [0, 25, 50, 75, 100])
    
    plt.subplot(2, 1, 2)
    acc_bin_base = [sum(bin) / len(bin) for bin in bins_base if len(bin) > 0]
    acc_bin_lba = [sum(bin) / len(bin) for bin in bins_lba if len(bin) > 0]
    plt.plot([i for i in range(len(acc_bin_base))], acc_bin_base, color='r')
    plt.plot([i for i in range(len(acc_bin_lba))], acc_bin_lba, color='g')
    # acc_bin_base = acc_bin_base[:cfg.runner_cfg.num_bin]
    # plt.plot([i+0.5 for i in range(len(acc_bin_base))], acc_bin_base, color='r')
    # json.dump(acc_bin_base, open(os.path.join(output_dir, f'vis/{cfg.datasets_cfg.dataset_name}_BLIP2.json'), 'w'))
    
    plt.title(f'{cfg.datasets_cfg.dataset_name} | acc for {cfg.runner_cfg.num_bin} bins by conf') # len(acc_bin)
    plt.xlabel('Confidence Percentile')
    plt.ylabel('Accuracy (Base: red, LBA: green)')
    plt.xticks([(cfg.runner_cfg.num_bin // 10) * i for i in range(11)], list(range(0, 101, 10)))
    
    # plt.xticks([(cfg.runner_cfg.num_bin // 5) * i for i in range(6)])
    plt.subplots_adjust(left=0.125, bottom=0.10, right=0.9, top=0.90, wspace=0.2, hspace=0.45)
    fig_path = os.path.join(output_dir, "acc_bin.png")
    plt.savefig(fig_path, dpi=300)
    print(f'saved fig path is {fig_path}')
    # draw heatmap
    draw_heatmap2(heatmap_data['number'], output_dir, 'number', H)
    draw_heatmap2(heatmap_data['change'], output_dir, 'change', H)
    draw_heatmap2(heatmap_data['change_abs'], output_dir, 'change_abs', H)
    
    # draw scatter plot
    plt.figure(figsize=(6,6))
    scatter_df = pd.DataFrame(scatter_data)#, columns=['conf_base', 'conf_lba', 'acc_change', 'size'])
    palette_params = {
        'w2w': 'gray',
        'w2r': 'green',
        'r2w': 'red',
        'r2r': 'black',
    }
    # color_map = {k: v['color'] for k, v in palette_params.items()}
    # label_map = {k: v['label'] for k, v in palette_params.items()}
    sns.scatterplot(x='conf_base', y='conf_lba', hue='class', data=scatter_df, 
                    palette=palette_params, #'rainbow', # 'coolwarm'
                    # hue_order=[1, 0, -1], 
                    size='size', sizes=(3, 7))
    # plt.legend(title='Class', labels=[label_map[i] for i in range(4)])
    plt.xlim(-5, 0)
    plt.ylim(-5, 0)
    plt.title(f'{cfg.datasets_cfg.dataset_name}')
    plt.xlabel('log Confidence score, Baseline')
    plt.ylabel('log Confidence score, LBA')
    fig_path = os.path.join(output_dir, "scatter.png")
    plt.savefig(fig_path, dpi=300)
    print(f'saved scatter fig path is {fig_path}')
    
    print(f'saved config path is {os.path.join(output_dir, "config.yaml")}')

    if cfg.runner_cfg.select_high_confidence:
        metrics = OrderedDict({
            "max_tau2          ": f'{max_conf_gap:.6g}',
            "I(max_tau2)       ": f'{np.log2(1/(max_conf_gap+1e-100)):.2f}',
            "acc_origin        ": f'{total_base_match / N * 100:.2f}%',
            "max_acc_by_tau    ": f'{max(final_acc_list) * 100:.2f}%',# = {max(final_acc_list)} / {N}',
        })
    else:
        metrics = OrderedDict({
            "acc_origin        ": f'{total_base_match / N * 100:.2f}% = {total_base_match} / {N}',
            "Previous_work_acc ": f'{baseline_max_match / N * 100:.2f}% = {baseline_max_match} / {N}',
        })
    
    if 'type' in results[0]:# or results[0]["question_id"][0] in "TCDISPFL": # cfg.datasets_cfg.dataset_name in ['DramaQA', 'NExTQA', 'STAR']:
        match_per_type = {}
        total_per_type = {}
        # total number of each question type
        # NExTQA : 4996 {'C': 2607, 'T': 1612, 'D': 777}
        # STAR   : 7098 {'Interaction': 2398, 'Sequence': 3586, 'Prediction': 624, 'Feasibility': 490}
        # DramaQA: 3889 { 'Level 2': 2698, 'Level 3': 1189}
        for i, result in enumerate(results):
            
            question_type = result['type']# if 'type' in results[0] else result["question_id"].split('_')[0]
            if cfg.datasets_cfg.dataset_name == 'NExTQA':
                question_type = question_type[0]
            
            target = result['gt_ans']
            
            # get predict
            if not cfg.runner_cfg.get("baseline", False) and result['confidence_base'] <= max_arg_confidence:
                if cfg.runner_cfg.select_high_confidence:
                    if result['confidence_base'] + max_conf_gap < result['confidence_lba']:
                        predict = result['text_output_lba']
                    else:
                        predict = result['text_output_base']
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
        # print("total_per_type:", total_per_type)
        
        for _q in "TCDISPFL":
            for q_type in match_per_type.keys():
                if q_type.startswith(_q) and total_per_type[q_type] > 0:
                    # qtype_v = f'{match_per_type[q_type] / total_per_type[q_type] * 100:4.2f}% = {match_per_type[q_type]:6.1f} / {total_per_type[q_type]:5d}'
                    qtype_v = f'{match_per_type[q_type] / total_per_type[q_type] * 100:4.2f}%'# = {match_per_type[q_type]} / {total_per_type[q_type]}'
                    # print(f'{q_type:<21s}: {qtype_v}')
                    metrics[f'{q_type:<18s}'] = qtype_v
    
    metrics.update({
        "max_arg_conf      ": f'{max_arg_confidence:.6f}',
        "conf_%            ": f'{confidence_percentile:.2f}%',
        "E_CR              ": f'{e_cr:.2f}%',
        "E_IC              ": f'{e_ic:.2f}%',
        # "min_match            ": f'{min_match / N * 100:.2f}%',
        "final_acc         ": f'{final_acc_list[-1] * 100:.2f}%',
    })
    
    print("metrics:", json.dumps(metrics, indent=4))

    # if not cfg.runner_cfg.visualize:
    with open(os.path.join(output_dir, "evaluate.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=4) + "\n")

    print('output_dir\t', 'dataset\t', 'recomposer\t')
    print('select_high_confidence\t', 'train_recomposer_examplar\t', 'vision_supple\t', 'use_pre_generated_sub_q\t')
    print('num_sub_qa_generate\t', 'num_sub_qa_select\t', 'pick_subq\t', 'max_conf_gap\t')
    print(saved_output_dir.split('/')[-1], end='\t')
    print(cfg.datasets_cfg.dataset_name, end='\t')
    print(cfg.runner_cfg.recomposer_name, end='\t')
    
    print(cfg.runner_cfg.get("select_high_confidence", False), end='\t')
    print(cfg.runner_cfg.get("train_recomposer_examplar", False), end='\t')
    print(cfg.runner_cfg.get("vision_supple", False), end='\t')
    print(cfg.runner_cfg.get("use_pre_generated_sub_q", False), end='\t')
    print(cfg.runner_cfg.get("num_sub_qa_generate", 1), end='\t')
    print(cfg.runner_cfg.get("num_sub_qa_select", 1), end='\t')
    print(cfg.runner_cfg.get("num_pick_subq", cfg.runner_cfg.get("num_sub_qa_generate", 1)), end='\t')
    print(f'{max_conf_gap:.5f}', end='\t')


    # print(f'copy and paste: {total_base_match / N * 100:.2f}\t{max(final_acc_list) * 100:.2f}\t{max_arg_confidence:.6f}\t{confidence_percentile:.2f}\t{e_cr:.2f}\t{e_ic:.2f}', end='')
    # if 'type' in results[0]:
    #     for _q in "TCDISPFL":
    #         for q_type in match_per_type.keys():
    #             if q_type.startswith(_q) and total_per_type[q_type] > 0:
    #                 print(f'\t{match_per_type[q_type] / total_per_type[q_type] * 100:.2f}', end='')
    print('\n')
    
    return metrics


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
    if key == 'change_abs':
        plt.xlabel('Base Confidence')
        plt.ylabel('LBA Confidence')
        plt.xticks(range(0, H+1, 2), [round(i-H//2) for i in range(H//2+1)])
        plt.yticks(range(0, H+1, 2), [round(-i) for i in range(H//2+1)])
        plt.title(f'Base -> LBA {key}')
    else:
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
    
    return round(float(b < l)), round(float(b > l)), round(b) == round(l) == 0, round(b) == round(l) == 1,  # w2r, r2w, w, r
    