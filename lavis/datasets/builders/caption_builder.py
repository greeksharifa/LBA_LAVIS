"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)
# from lavis.datasets.datasets.vqa_introspect_caption_datasets import (
#     VQAIntrospectCapDataset,
#     VQAIntrospectCapEvalDataset
# )
# from lavis.datasets.datasets.vqa_introspect_multiple_caption_datasets import (
#     VQAIntrospectMultipleCapDataset,
#     VQAIntrospectMultipleCapEvalDataset
# )
from lavis.datasets.datasets.vqa_introspect_datasets import (
    VQAIntrospectQARCapDataset,
    VQAIntrospectQARCapEvalDataset
)


@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }

#
# @registry.register_builder("vqa_introspect_caption")
# class VQAIntrospectCapBuilder(BaseDatasetBuilder):
#     # print('in VQAIntrospectCapBuilder class')
#     train_dataset_cls = VQAIntrospectCapDataset
#     eval_dataset_cls = VQAIntrospectCapEvalDataset
#
#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/vqa_introspect/defaults_cap.yaml",
#     }
#
#
# @registry.register_builder("vqa_introspect_multiple_caption")
# class VQAIntrospectMultipleCapBuilder(BaseDatasetBuilder):
#     # print('in VQAIntrospectMultipleCapBuilder class')
#     train_dataset_cls = VQAIntrospectMultipleCapDataset
#     eval_dataset_cls = VQAIntrospectMultipleCapEvalDataset
#
#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/vqa_introspect/defaults_multiple_cap.yaml",
#     }


@registry.register_builder("vqa_introspect_qar_caption")
class VQAIntrospectQARCapBuilder(BaseDatasetBuilder):
    # print('in VQAIntrospectMultipleCapBuilder class')
    train_dataset_cls = VQAIntrospectQARCapDataset
    eval_dataset_cls = VQAIntrospectQARCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa_introspect/defaults.yaml",
    }
