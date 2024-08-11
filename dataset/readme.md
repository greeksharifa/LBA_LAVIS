


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