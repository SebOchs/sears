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
from preprocessing_bert import BertPreprocessor

# TextFooler imports
sys.path.append('../TextFooler')
import criteria
import tensorflow

tf = tensorflow.compat.v1
tf.disable_eager_execution()
import tensorflow_hub as hub

# Own imports
from transformers import BertForSequenceClassification, BertTokenizer

import torch.nn.functional as F

nlp = en_core_web_md.load()
tokenizer = replace_rules.Tokenizer(nlp)
pretrained_weights = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


def list_to_string(list):
    """
    joins a list of strings together
    :param list: list of strings
    :return: string
    """
    return ' '.join(list)


def separate_answers(bert_text, cls='[CLS]', sep='[SEP]'):
    """
    Separates the sentences of sequence classification used for bert
    :param bert_text: list of bert word tokens
    :param cls: string of cls token
    :param sep: string of sep token
    :return: separated strings
    """
    # Fix double-hash
    pattern = r'^##.*'
    remove = []
    for i, word in enumerate(bert_text):
        if re.match(pattern, word):
            bert_text[i] = bert_text[i - 1] + word[2:]
            remove.append(i - 1)
    for j in sorted(remove, reverse=True):
        bert_text.pop(j)
    cls_idx = bert_text.index(cls)
    sep_1_idx = bert_text.index(sep)
    ans1 = bert_text[cls_idx + 1:sep_1_idx]
    ans2 = bert_text[sep_1_idx + 1:bert_text.index(sep, sep_1_idx + 1)]
    return ans1, ans2


def predict(model, ref, stud, orig_pred):
    if type(ref) is list:
        ref = list_to_string(ref)
    if type(stud) is list:
        stud = list_to_string(stud)
    assert type(stud) is str and type(ref) is str
    token_ids, segment, attention, lab = \
        BertPreprocessor(bert_tokenizer, data=[ref, stud, orig_pred]).load_data()
    token_ids = torch.tensor([token_ids]).long().to(device)
    segment = torch.tensor([segment]).long().to(device)
    attention = torch.tensor([attention]).long().to(device)
    outputs = model.forward(input_ids=token_ids, attention_mask=attention, token_type_ids=segment)
    logits = outputs[0].detach().cpu().squeeze()
    return logits


def apply_rules(dataset_path, model_path, top_rules):

    data = np.load(dataset_path, allow_pickle=True)
    # Some reference answers are too long and wrongly cut off (9 cases out of 8205)
    data = [x for x in data if x[0].count('[SEP]') > 1]
    test_data = [separate_answers(x[0]) for x in data if x[1] == 0]
    tr2 = replace_rules.TextToReplaceRules(nlp, [list_to_string(x[1]) for x in test_data], [], min_freq=0.005,
                                           min_flip=0.005, ngram_size=2)
    # Own model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    tokenized_stud_ans = tokenizer.tokenize([list_to_string(x[1]) for x in test_data])
    model_preds = {}
    data_flip_amount = {}
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
                reference_answer = list_to_string(test_data[j][0])
                logits = predict(model, reference_answer, new_stud, 0)
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
    top_rules = np.load("bs_sears_final_rules.npy", allow_pickle=True)
    a = time.time()
    apply_rules('../bachelor-thesis/data/eval_data/bert_seb_ua_correct.npy',
                '../bachelor-thesis/models/bert_sciEntsBank/model.pt', top_rules)

    print("Done with SEB UA")
    apply_rules('../bachelor-thesis/data/eval_data/bert_seb_uq_correct.npy',
                '../bachelor-thesis/models/bert_sciEntsBank/model.pt', top_rules)

    print("Done with SEB UQ")
    apply_rules('../bachelor-thesis/data/eval_data/bert_seb_ud_correct.npy',
                '../bachelor-thesis/models/bert_sciEntsBank/model.pt', top_rules)

    print("Done with SEB UD")
    apply_rules('../bachelor-thesis/data/eval_data/bert_bee_ua_correct.npy',
                '../bachelor-thesis/models/bert_beetle/model.pt', top_rules)

    print("Done with BEE UA")
    apply_rules('../bachelor-thesis/data/eval_data/bert_bee_uq_correct.npy',
                '../bachelor-thesis/models/bert_beetle/model.pt', top_rules)

    print("Done with BEE UQ")
    apply_rules('../bachelor-thesis/data/eval_data/bert_mnli_m_correct.npy',
                '../bachelor-thesis/models/bert_mnli/model_mnli.pt', top_rules)

    print("Done with matched")
    apply_rules('../bachelor-thesis/data/eval_data/bert_mnli_mm_correct.npy',
                '../bachelor-thesis/models/bert_mnli/model_mnli.pt', top_rules)
    print("runtime: ", time.time() - a)
    print("Done with mismatched")



if __name__ == "__main__":
    main()
