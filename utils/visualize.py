import json
import os
import matplotlib.pyplot as plt


def visualize(results, dataset, cfg, output_dir, total_base_match):
    results.sort(key=lambda x: x['confidence_lba'])
    
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
        acc_lba = dataset.get_accuracy(result['text_output_lba'], result['gt_ans'], match1ok=cfg.runner_cfg.match1ok)
        acc_base_list.append(acc_base)
        acc_lba_list.append(acc_lba)
        
        bin_key = i // M
        bins[bin_key].append(acc_base)
        
        cur_match += acc_lba - acc_base
        match_list.append(cur_match)
        
        if cur_match > max_match:
            max_match = cur_match
            max_arg_confidence = result['confidence_base']
            confidence_percentile = (i+1) / N * 100
            
    final_acc_list = [match / N for match in match_list]
    
    
    # E_CR, E_IC: Error Correction raio / Error Induction ratio
    e_cr, e_ic = dataset.get_e_cr_e_ic(acc_base_list, acc_lba_list)
    
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    # plt.subplots(constrained_layout=True)
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
    plt.savefig(fig_path, dpi=300)
    print(f'saved fig at {fig_path}')
    
    metrics = {
        "acc_origin           ": f'{total_base_match / N * 100:.3f}%',
        "max_acc_by_tau       ": f'{max(final_acc_list) * 100:.3f}%', 
        "max_arg_confidence   ": f'{max_arg_confidence:.6f}',
        "confidence_percentile": f'{confidence_percentile:.3f}%',
        "E_CR                 ": f'{e_cr:.2f}%',
        "E_IC                 ": f'{e_ic:.2f}%',
    }
    print("metrics:", json.dumps(metrics, indent=4))

    with open(os.path.join(output_dir, "evaluate.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=4) + "\n")
        