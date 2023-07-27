import pandas as pd
import json

data = pd.read_json('data/VQA-Introspect/VQAIntrospect_trainv1.0.json')

json_data = json.load(open('data/VQA-Introspect/VQAIntrospect_valv1.0.json', "r"))
json_data = json.load(open('data/VQA-Introspect/VQAIntrospect_trainv1.0.json', "r"))

for question_id, value in json_data.items():
    image_id = value["image_id"]
    main_question = value["reasoning_question"]
    main_answer = value["reasoning_answer_most_common"]
    
    for introspect in value["introspect"]:
        sub_qa_list = introspect["sub_qa"]
        pred_q_type = introspect["pred_q_type"]
        
        for sub_qa in sub_qa_list:
            _sample = {
                "image_id": image_id,
                "question_id": question_id,
                "main_question": main_question,
                "main_answer": main_answer,
                "sub_question": sub_qa["sub_question"],
                "sub_answer": sub_qa["sub_answer"],
                "pred_q_type": pred_q_type,
            }
            if sub_qa["sub_answer"] == 'zebra':
                print(_sample)
                