


# NExTQA
```python
# total 4996 
# {'C': 2607, 'T': 1612, 'D': 777}
{
    'num_option': 5, 
    'qid'       : 'TN_6233408665_8', 
    'question'  : 'what did the people on the sofa do after the lady in pink finished singing?', 
    'video'     : '1150/6233408665',
    'a0'        : 'sitting.', 
    'a1'        : 'give it to the girl.', 
    'a2'        : 'take music sheet.', 
    'a3'        : 'clap.', 
    'a4'        : 'walk in circles.', 
    'answer'    : 3
}
```


# STAR
```python
# total 7098
# {'Interaction': 2398, 'Sequence': 3586, 'Prediction': 624, 'Feasibility': 490}
# dict_keys(['question_id', 'question', 'video_id', 'start', 'end', 'answer', 'question_program', 'choices', 'situations'])
{
    'answer'    : 'The clothes.',
    'choices'   : [{'choice': 'The closet/cabinet.', 'choice_id': 0, 'choice_program': [{'function': 'Equal', 'value_input': ['closet/cabinet']}]},
                   {'choice': 'The blanket.', 'choice_id': 1, 'choice_program': [{'function': 'Equal', 'value_input': ['blanket']}]},
                   {'choice': 'The clothes.', 'choice_id': 2, 'choice_program': [{'function': 'Equal', 'value_input': ['clothes']}]},
                   {'choice': 'The table.', 'choice_id': 3, 'choice_program': [{'function': 'Equal', 'value_input': ['table']}]}],
    'end'       : 19.6,
    'question'  : 'Which object was tidied up by the person?',
    'question_id': 'Interaction_T1_13',
    'question_program': [...],
    'situations': {...},
    'start'     : 11.1,
    'video_id'  : '6H78U'
 }
```


# TVQA
```python
# total 15253
{
    'a0': 'Because Sheldon is being rude.',
    'a1': "Because he doesn't like Sheldon.",
    'a2': 'Because they are having an argument.',
    'a3': 'Because Howard wanted to have a private meal with Raj.',
    'a4': "Because Sheldon won't loan him money for food.",
    'answer': 2,
    'end': '25.12',
    'num_option': 5,
    'qid': 'TVQA_0',
    'question': 'Why is Howard frustrated when he is talking to Sheldon?',
    'start': '20.16',
    'video': 's03e02_seg02_clip_10'
}
```


# VLEP
```python
# total 4392
{
    'a0': 'Ross will stop, turn and point at Monica.',
    'a1': 'Ross will stop and ask Monica why she is pointing at him.',
    'answer': 0,
    'end': 40.37,
    'num_option': 2,
    'qid': 'VLEP_20142',
    'start': 38.81,
    'video': 'friends_s03e09_seg02_clip_07_ep'
}
```