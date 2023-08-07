# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
import copy
from omegaconf import open_dict
from dataclasses import dataclass, field
import pickle
import os
import sys
import time


@dataclass
class LabelSmoothedDualImitationCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("SCNAT_criterion", dataclass=LabelSmoothedDualImitationCriterionConfig)
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(
                    logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(
                    logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) -
                    mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def load_translation_dict(self, PATH):

        if os.path.exists(PATH):
            with open(PATH, "rb") as f:

                l = pickle.load(f)

                return l
        return {}

    def forward(self, model, sample, pos_model, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        #### POSDP ####

        translation_dict_index = 0
        PATH = "./data-bin/IWSLT_EN_VT/translation_dicts/"

        all_translation_dicts = os.listdir(PATH)

        if len(all_translation_dicts) == 0:

            translation_dict_index = 0
            model.translation_dict = {}
            with open(PATH + f"translation_dict_{translation_dict_index}.pkl", "wb") as f:
                pickle.dump(model.translation_dict, f)

            all_translation_dicts = os.listdir(PATH)

        device = src_tokens.device
        src_sentences = [model.encoder.dictionary.string(
            src_token) for src_token in src_tokens]

        # check which sentences are in dict
        tokenized_sentences = [pos_model.encode(
            sentence) for sentence in src_sentences]

        known_sentences = torch.zeros(
            src_tokens.shape[0]).bool().to(device)

        known_dict = {}

        start_time = time.time()
        for translation_dict_index, translation_dict in enumerate(all_translation_dicts):

            model.translation_dict = self.load_translation_dict(
                PATH + translation_dict)

            for i, src_sentence in enumerate(src_sentences):
                if src_sentence in model.translation_dict:
                    known_sentences[i] = True
                    pos_tag, pos_id = model.translation_dict[src_sentence]
                    known_dict[src_sentence] = (pos_tag, pos_id)

            if known_sentences.all():
                break

        print(f"LOADED {int(time.time() - start_time)} seconds")

        if len(model.translation_dict.keys()) >= 1500 and not known_sentences.all():
            translation_dict_index += 1
            model.translation_dict = {}

        # translate unknown sentences
        unknown_sentences = ~known_sentences

        if unknown_sentences.any():

            sentences_to_translate = [
                tokenized_sentences[i] for i, unknown_sentence in enumerate(unknown_sentences) if unknown_sentence]
            # pos_tags = pos_model.translate(src_sentences)

            # build generator using current args as well as any kwargs
            gen_args = copy.deepcopy(pos_model.cfg.generation)
            with open_dict(gen_args):
                gen_args.beam = 5
                # for k, v in kwargs.items():
                #     setattr(gen_args, k, v)
            generator = pos_model.task.build_generator(
                pos_model.models,
                gen_args,
                prefix_allowed_tokens_fn=None,
            )

            inference_step_args = {}
            results = []
            for batch in pos_model._build_batches(sentences_to_translate, False):
                batch = utils.apply_to_sample(
                    lambda t: t.to(pos_model.device), batch)
                translations = pos_model.task.inference_step(
                    generator, pos_model.models, batch, **inference_step_args
                )
                for id, hypos in zip(batch["id"].tolist(), translations):
                    results.append((id, hypos))

            # sort output to match input order

            pos_out = [hypos for _, hypos in sorted(
                results, key=lambda x: x[0])]

            pos_ids = [hypos[0]["tokens"]
                       for hypos in pos_out]

            pos_tags = [pos_model.decode(hypos[0]["tokens"])
                        for hypos in pos_out]

            lower_limit = 0
            for i, src_sentence in enumerate(src_sentences):
                if unknown_sentences[i]:
                    model.translation_dict[src_sentences[i]] = (
                        pos_tags[lower_limit], pos_ids[lower_limit])

                    lower_limit += 1

            with open(PATH + f"translation_dict_{translation_dict_index}.pkl", "wb") as f:
                print("UNSAFE SAVING")
                pickle.dump(model.translation_dict, f)
                print(
                    f"SAVED {PATH + f'translation_dict_{translation_dict_index}.pkl'}")

            del gen_args
            del generator

        pos_tags = []
        pos_ids = []

        precentage_known = 0
        precentage_unknown = 0

        for i, src_sentence in enumerate(src_sentences):

            if src_sentence in known_dict:
                pos_tag, pos_id = known_dict[src_sentence]
                pos_tags.append(pos_tag)
                pos_ids.append(pos_id)
                precentage_known += 1

            else:
                pos_tag, pos_id = model.translation_dict[src_sentence]
                pos_tags.append(pos_tag)
                pos_ids.append(pos_id)
                precentage_unknown += 1

        print(
            f"Known: {precentage_known / len(src_sentences) * 100} % Unknown: {precentage_unknown / len(src_sentences) * 100} %")

        del model.translation_dict
        del known_dict

        pos_ids = torch.zeros(
            (len(pos_ids), max([len(ids) for ids in pos_ids]))).long().to(device)

        for i, ids in enumerate(pos_ids):
            pos_ids[i, :len(ids)] = ids

        pos_tag_sentences = []

        for pos_tag_sentence in pos_tags:
            sentence = []
            for pos_tag in pos_tag_sentence.split(" "):

                sentence.append(pos_tag)

            pos_tag_sentences.append(sentence)

        length_tgt = torch.tensor([len(pos_tag_sentence)
                                   for pos_tag_sentence in pos_tag_sentences]).to(device)
        #### POSDP ####

        outputs = model(src_tokens, src_lengths,
                        prev_output_tokens, tgt_tokens, pos_ids=pos_ids)
        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(
            nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0)
                                  for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size /
                    math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
