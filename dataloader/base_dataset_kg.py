import torch
from torch.utils.data import Dataset
import copy

class BaseDataset_kg(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.max_feats = args.max_feats
        self.features_dim = 768
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.split = split

    def _get_padding_id(
        self,
        text_id,
        prefix_index,
        context_start,
        context_end,
    ):
        padding_text_id = torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        overflow = any(len(tid) > self.max_seq_len for tid in text_id)
        context_keep = context_end - context_start
        adjusted_prefix = prefix_index
        if overflow:
            max_suffix_length = max(len(tid) - context_end for tid in text_id)
            context_keep = self.max_seq_len - context_start - max_suffix_length
            if context_keep < 0:
                raise ValueError(
                    "max_seq_len is too short to preserve the KG prompt suffix"
                )
            adjusted_prefix -= context_end - context_start - context_keep
            if getattr(self.args, "rank", 0) == 0:
                print('max sequence length overflow')

        for i, tid in enumerate(text_id):
            if overflow:
                cropped = torch.cat(
                    (
                        tid[:context_start],
                        tid[context_start:context_start + context_keep],
                        tid[context_end:],
                    )
                )
            else:
                cropped = tid
            padding_text_id[i, :len(cropped)] = cropped
        return padding_text_id, adjusted_prefix

    def _get_text_token(self, text, answer):
        vqa_id, vqa_prefix_index, vqa_video_start, vqa_context_start, vqa_context_end = self.tokenizer.encode_kvqa(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vaq_id, vaq_prefix_index, vaq_video_start, vaq_context_start, vaq_context_end = self.tokenizer.encode_kvaq(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        qav_id, qav_prefix_index, qav_context_start, qav_context_end = self.tokenizer.encode_kqav(text=text, max_feats=self.max_feats, max_seq_len=self.max_seq_len, split=self.split, answer_mapping=self.answer_mapping, answer=answer)

        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vaq_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vaq_id]
        qav_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in qav_id]

        vqa_padding_text_id, vqa_prefix_index = self._get_padding_id(
            vqa_id,
            vqa_prefix_index,
            vqa_context_start,
            vqa_context_end,
        )
        vaq_padding_text_id, vaq_prefix_index = self._get_padding_id(
            vaq_id,
            vaq_prefix_index,
            vaq_context_start,
            vaq_context_end,
        )
        qav_padding_text_id, qav_prefix_index = self._get_padding_id(
            qav_id,
            qav_prefix_index,
            qav_context_start,
            qav_context_end,
        )

        # label
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label_mask = vqa_label.ge(0)
        vqa_label[~vqa_label_mask] = 0
        vqa_label_mask = vqa_label_mask.float()

        vaq_label = copy.deepcopy(vaq_padding_text_id)
        vaq_label[:, :vaq_prefix_index] = -1
        vaq_label_mask = vaq_label.ge(0)
        vaq_label[~vaq_label_mask] = 0
        vaq_label_mask = vaq_label_mask.float()

        qav_label = torch.ones_like(qav_padding_text_id) * -1
        qav_label[:, qav_prefix_index:qav_prefix_index+self.max_feats] = torch.arange(self.max_feats)
        qav_label_mask = torch.zeros_like(qav_padding_text_id)
        qav_label_mask[:, qav_prefix_index] = 1
        qav_label_mask = qav_label_mask.float()

        # text mask
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
        vaq_text_mask = vaq_padding_text_id.ge(0)
        vaq_padding_text_id[~vaq_text_mask] = 0
        qav_text_mask = qav_padding_text_id.ge(0)
        qav_padding_text_id[~qav_text_mask] = 0

        # video index
        vqa_video_index = torch.arange(vqa_prefix_index, vqa_prefix_index + self.max_feats)
        vaq_video_index = torch.arange(vaq_prefix_index, vaq_prefix_index + self.max_feats)
        qav_video_index = torch.arange(qav_prefix_index, qav_prefix_index + self.max_feats)


        text_id = {'vqa': vqa_padding_text_id, 'vaq': vaq_padding_text_id, 'qav': qav_padding_text_id}
        label = {'vqa': vqa_label, 'vaq': vaq_label, 'qav': qav_label}
        video_start = {'vqa': vqa_video_start, 'vaq': vaq_video_start, 'qav': qav_prefix_index}
        video_index = {'vqa': vqa_video_index, 'vaq': vaq_video_index, 'qav': qav_video_index}
        label_mask = {'vqa': vqa_label_mask, 'vaq': vaq_label_mask, 'qav': qav_label_mask}
        return text_id, label, video_start, video_index, label_mask
