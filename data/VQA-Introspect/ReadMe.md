{%dictionary of reasoning question ids
<reas_qid>:
{
‘reasoning_question’:<reasoning_question_from_vqa_dataset>,
‘reasoning_answer_most_common’:<reasoning_answer_from_vqa_dataset>, 
‘introspect’: 
[ %list of responses from different turkers % multiple turkers were asked to provide subquestions
{ %worker 1
‘pred_q_type’: <pred_mainq_type_from_turker>, %workers categorized the type of main-question <reasoning, perception, invalid >
’sub_qa’: 
[%list of subquestion answers from the same worker (note that workers could have given multiple sub-questions for a single <mainq, maina> pair
{%subquestion answer 1
‘subquestion’: ‘<subquestion_from_our_collected_dataset>’, ‘answer’:’<subanswer_from_our_collected_dataset>’ 
},
{%subquestion answer 2
‘subquestion’: ‘<subquestion_from_our_collected_dataset>’, ‘answer’:’<subanswer_from_our_collected_dataset>’ 
},
{%subquestion answer n}
...
]

}
{%worker 2} ,
{%worker n}
]
}
} 
