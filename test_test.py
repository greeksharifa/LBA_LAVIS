import json

"""base
[
    {
        "image_id": 565248,
        "question_id": "565248001",
        "question": "are they at the beach?",
        "pred_ans": "yes",
        "gt_ans": "yes"
    },
    ...
]
"""

"""ours
    {
        "image_id": 565248,
        "question_id": "565248001",
        "question": "are they at the beach?",
        "pred_ans": "yes",
        "gt_ans": "yes",
        "sub_q_list": "is there sand and water?",
        "sub_a_list": "yes"
    },

"""


base = json.load(open("lavis/ywjang_output_qar_test/20230815093/result/val_vqa_result.json", "r"))
# ours = json.load(open("lavis/ywjang_output_qar_aok_vqa/20230815101/result/val_vqa_result.json", "r")) # SQ 1
ours = json.load(open("lavis/ywjang_output_qar_aok_vqa/20230815103/result/val_vqa_result.json", "r")) # SQ 2

for b, o in zip(base, ours):
    if b["pred_ans"] != b["gt_ans"] and o["pred_ans"] == o["gt_ans"]:
        print('ID:', b["image_id"])
        print('MQ:', b["question"])
        print('SQ:', o["sub_q_list"])
        print('SA:', o["sub_a_list"])
        print('MA:', b["gt_ans"])
        print('BA:', b["pred_ans"])
        print('OA:', o["pred_ans"])
        print()
        
print('done')
