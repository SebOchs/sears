# Script to apply rules to a given test set using a given model to predict
# Rules have been generated to flip incorrect student answers to correct
# Using MNLI, the neutral sequences should be flipped to entailment
# GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

device = torch.device("cuda")

# Main imports
import collections
import en_core_web_md
import itertools
import numpy as np
import os
import random
import re
import spacy
import sys
import time

# Extern imports
sys.path.append('sears')  # noqa
from sears import replace_rules

sys.path.append('../bachelor-thesis')
sys.path.append('../bachelor-thesis/bert')
from preprocessing_albert import AlbertPreprocessor

# TextFooler imports
sys.path.append('../TextFooler')
import criteria
import tensorflow

tf = tensorflow.compat.v1
tf.disable_eager_execution()
import tensorflow_hub as hub

# Own imports
from transformers import AlbertForSequenceClassification, AlbertTokenizer

import torch.nn.functional as F

nlp = en_core_web_md.load()
tokenizer = replace_rules.Tokenizer(nlp)
pretrained_weights = 'albert-base-v1'
albert_tokenizer = AlbertTokenizer.from_pretrained(pretrained_weights)


def list_to_string(list):
    """
    joins a list of strings together
    :param list: list of strings
    :return: string
    """
    return ' '.join(list)


def separate_answers(albert_text, cls='[CLS]', sep='[SEP]'):
    """
    Separates the sentences of sequence classification used for bert
    :param bert_text: list of bert word tokens
    :param cls: string of cls token
    :param sep: string of sep token
    :return: separated strings
    """
    # Fix SPIECE_underline
    cls_idx = albert_text.index(cls) + 4
    sep_1_idx = albert_text.index(sep) + 4
    ans1 = albert_text[cls_idx + 1:sep_1_idx - 4]
    ans2 = albert_text[sep_1_idx + 1:albert_text.index(sep, sep_1_idx + 1)]
    return ans1, ans2


def predict(model, ref, stud, orig_pred):
    if type(ref) is list:
        ref = list_to_string(ref)
    if type(stud) is list:
        stud = list_to_string(stud)
    assert type(stud) is str and type(ref) is str
    token_ids, segment, attention, lab = \
        AlbertPreprocessor(albert_tokenizer, data=[ref, stud, orig_pred]).load_data()
    token_ids = torch.tensor([token_ids]).long().to(device)
    segment = torch.tensor([segment]).long().to(device)
    attention = torch.tensor([attention]).long().to(device)
    outputs = model.forward(input_ids=token_ids, attention_mask=attention, token_type_ids=segment)
    logits = outputs[0].detach().cpu().squeeze()
    return logits


def apply_rules(dataset_path, model_path):
    data = np.load(dataset_path, allow_pickle=True)

    test_data = [separate_answers(x[0]) for x in data if int(x[1]) == 0]
    top_rules = np.load("bs_sears_final_rules.npy", allow_pickle=True)
    tr2 = replace_rules.TextToReplaceRules(nlp, [x[1] for x in test_data], [], min_freq=0.005,
                                           min_flip=0.005, ngram_size=2)
    # Own model
    model = AlbertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=3)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    tokenized_stud_ans = tokenizer.tokenize([x[1] for x in test_data])
    model_preds = {}
    rule_flip_amount = {}
    data_id_flipped = {}
    a = time.time()
    for rule in top_rules:
        idxs = list(tr2.get_rule_idxs(rule))
        to_apply = [tokenized_stud_ans[x] for x in idxs]
        applies, nt = rule.apply_to_texts(to_apply, fix_apostrophe=False)
        # Find indices, where rule has been applied
        applies = [idxs[x] for x in applies]
        to_compute = [x for x in zip(applies, nt) if x[1] not in model_preds]
        if to_compute:
            # New predicts
            new_labels = []
            for compute in to_compute:
                j, new_stud = compute
                # Get reference answer for sequence classification
                orig_instance = test_data[j]
                logits = predict(model, orig_instance[0], new_stud, 0)
                new_label = int(np.argmax(logits))
                new_labels.append(new_label)
            for x, y in zip(to_compute, new_labels):
                model_preds[x[1]] = y

        new_labels = np.array([model_preds[x] for x in nt])
        where_flipped = np.where(new_labels == 2)[0]
        flips = sorted([applies[x] for x in where_flipped])
        rule_flip_amount[rule.hash()] = len(flips)
        data_id_flipped[rule.hash()] = list(where_flipped)

        #print("Done with " + rule.hash())

    # Top 10 rules
    top_10 = [x.replace("text_", "").replace("pos_", "") for x in
              list({k: v for k, v in sorted(rule_flip_amount.items(), key=lambda item: item[1], reverse=True)})[:10]]
    np.save(model_path[:model_path.rfind("/") + 1] + "bs_top_10.npy", top_10)
    print("Time used for applying rules: ", time.time() - a)
    print("Total amount of adversaries:", sum(list(rule_flip_amount.values())))
    print("Total amount of afflicted data instances:",
          len(set(np.concatenate(list(data_id_flipped.values())).ravel().tolist())))


def main():
    apply_rules('../bachelor-thesis/data/eval_data/albert_seb_ua_test_correct.npy',
                '../bachelor-thesis/models/albert_sciEntsBank/albert_model_sciEntsBank.pt')
    print("Done with UA SEB")
    apply_rules('../bachelor-thesis/data/eval_data/albert_seb_uq_test_correct.npy',
                '../bachelor-thesis/models/albert_sciEntsBank/albert_model_sciEntsBank.pt')
    print("Done with UQ SEB")
    apply_rules('../bachelor-thesis/data/eval_data/albert_seb_ud_test_correct.npy',
                '../bachelor-thesis/models/albert_sciEntsBank/albert_model_sciEntsBank.pt')
    print("Done with UD SEB")
    apply_rules('../bachelor-thesis/data/eval_data/albert_bee_ua_test_correct.npy',
                '../bachelor-thesis/models/albert_beetle/albert_model.pt')
    print("Done with UA BEE")
    apply_rules('../bachelor-thesis/data/eval_data/albert_bee_uq_test_correct.npy',
                '../bachelor-thesis/models/albert_beetle/albert_model.pt')
    print("Done with UQ BEE")



if __name__ == "__main__":
    main()
print("Done.")
