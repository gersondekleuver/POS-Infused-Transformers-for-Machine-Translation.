# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    load_langpair_dataset,
)
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from typing import Any, Dict, List, Optional
from .POSDP_translation import *
from fairseq.utils import new_arange


NOISE_CHOICES = ChoiceEnum(
    ["random_delete", "random_mask", "no_noise", "full_mask"])


@dataclass
class POSDPTranslationLevenshteinConfig(POSDPTranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={"help": "type of noise"},
    )
    # pos_model: str = field(
    #     default="checkpoints/wmt14_transformer_pos_v2/",
    #     metadata={"help": "path to POS model"},
    # )
    pos_model: str = field(
        default="checkpoints/IWSLT_EN_VT/",
        metadata={"help": "path to POS model"},
    )
    checkpoint_file: str = field(
        default="checkpoint_best.pt",
        metadata={"help": "path to POS model"},
    )
    # data_name_or_path: str = field(
    #     default="data-bin/wmt14_en_de/pos",
    #     metadata={"help": "path to POS model"},
    # )
    # ldata: str = field(
    #     default="data-bin/wmt14_en_de/lang",
    #     metadata={"help": "path to POS model"},
    # )
    # pos_data: str = field(
    #     default="data-bin/wmt14_en_de/pos",
    #     metadata={"help": "path to POS model"},
    # )
    data_name_or_path: str = field(
        default="data-bin/IWSLT_EN_VT/pos",
        metadata={"help": "path to POS model"},
    )
    ldata: str = field(
        default="data-bin/IWSLT_EN_VT/lang",
        metadata={"help": "path to POS model"},
    )
    pos_data: str = field(
        default="data-bin/IWSLT_EN_VT/pos",
        metadata={"help": "path to POS model"},
    )
    pos_task: str = field(
        default="pos_translation",
        metadata={"help": "path to POS model"},
    )
    penalty: float = field(
        default=50.00,
        metadata={"help": "pos penalty"},
    )


@register_task("POSDP_task", dataclass=POSDPTranslationLevenshteinConfig)
class TranslationLevenshteinTask_v2(POSDP_translation):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    Added POS tagging model initialization and forward pass.
    """
    cfg: POSDPTranslationLevenshteinConfig

    def __init__(self, cfg: POSDPTranslationLevenshteinConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        if cfg.pos_model is not None:
            self.pos_model = PosTransformer.from_pretrained(
                cfg.pos_model,
                checkpoint_file=cfg.checkpoint_file,
                source_lang=cfg.source_lang,
                target_lang=cfg.target_lang,
                data_name_or_path=cfg.data_name_or_path,
                ldata=cfg.ldata,
                data=cfg.pos_data,
                task=cfg.pos_task,
            )

        else:
            raise "No POS model provided"
            raise NotImplementedError

        self.pos_model.penalty = cfg.penalty

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(
                    bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            # make sure to mask at least one token.
            target_length = target_length + 1

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(
                target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(
                    eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from ..models.POSDP.iterative_refinement_generator_POSDP import IterativeRefinementGenerator

        x = IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(
                args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )
        x.models = models[0].pos_model = self.pos_model
        return x

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()

        # feed the POS tags to the decoder
        sample["prev_target"] = self.inject_noise(sample["target"])

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()

        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
