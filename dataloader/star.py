import torch
from .base_dataset import BaseDataset
from .inquirer_paths import resolve_dataset_path, resolve_inquirer_path
import json
import random

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        source_root = getattr(args, "inquirer_source_root", None)
        dataset_root = getattr(args, "star_dataset_root", None)
        if split == 'train':
            if args.naive == True:
                original_data = json.loads(
                    resolve_dataset_path(
                        "star",
                        f"STAR_{split}_ori.json",
                        dataset_root=dataset_root,
                    ).read_text()
                )
                total_naive_data = json.loads(
                    resolve_inquirer_path(
                        "star",
                        f"STAR_{split}_naive_prob_filtered.json",
                        source_root=source_root,
                    ).read_text()
                )
                #additional_data.sort(key=lambda x: x['perplex']) # Descending
                #additional_data.sort(key=lambda x: x['perplex'], reverse=True) # Ascending
                #r = args.add_filter_ratio

                #self.data = original_data + additional_data[int(len(additional_data) * r):] # Ascending
                #self.data = original_data + additional_data[:int(len(additional_data) * r)] # Descending
                # 87.5% = 5625, 75% = 11250, 50% = 22500
                r = args.add_filter_ratio
                len_total_naive_data = int(len(total_naive_data) * r)
                total_naive_data = random.sample(total_naive_data, len_total_naive_data)
                #total_naive_data = random.sample(total_naive_data, args.naive_num)
                self.data = original_data + total_naive_data
            else:
                original_data = json.loads(
                    resolve_dataset_path(
                        "star",
                        f"STAR_{split}_ori.json",
                        dataset_root=dataset_root,
                    ).read_text()
                )
                additional_data = json.loads(
                    resolve_inquirer_path(
                        "star",
                        f"STAR_{split}_add_prob.json",
                        source_root=source_root,
                    ).read_text()
                )
                '''
                {
                    'question_id': 'Interaction_T1_4',
                    'video_id': 'TJZ0P',
                    'start': 7.7,
                    'end': 15.7,
                    'question': 'What type of object was present in the scene?',
                    'answer': 'A sandwich.',
                    'choices': [
                        {'choice_id': 0, 'choice': 'A sandwich.'},
                        {'choice_id': 1, 'choice': 'A chair.'},
                        {'choice_id': 2, 'choice': 'A book.'},
                        {'choice_id': 3, 'choice': 'A bottle.'}
                    ],
                    'q_type': 'Feature specification',
                    'perplex': 0.00020488160953391343
                }
                '''
                #additional_data.sort(key=lambda x: x['perplex']) # Descending
                additional_data.sort(key=lambda x: x['perplex'], reverse=True) # Ascending
                r = args.add_filter_ratio
                self.data = original_data + additional_data[int(len(additional_data) * r):]
                #self.data = original_data + additional_data[:int(len(additional_data) * r)]
        else:
            original_data = json.loads(
                resolve_dataset_path(
                    "star",
                    f"STAR_{split}_ori.json",
                    dataset_root=dataset_root,
                ).read_text()
            )
            self.data = original_data

        self.features = torch.load(
            resolve_dataset_path(
                "star",
                "clipvitl14.pth",
                dataset_root=dataset_root,
            )
        )
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
        self.num_options = 4
        if getattr(args, "rank", 0) == 0:
            print(f"Num {split} data: {len(self.data)}")


    def _get_text(self, idx):
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = {x['choice_id']: x['choice'] for x in self.data[idx]['choices']}
        options = [options[i] for i in range(self.num_options)]
        answer = options.index(self.data[idx]['answer'])

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id][start: end +1, :].float() # ts
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        qtype = self.qtype_mapping[self.data[idx]['question_id'].split('_')[0]]
        text, answer = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        start, end = round(self.data[idx]['start']), round(self.data[idx]['end'])
        video, video_len = self._get_video(f'{vid}', start, end)
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}


    def __len__(self):
        return len(self.data)
