import os

from transformers import BertForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from bert.preprocessing import BertPreprocessor

sys.path.append('sears')  # noqa
import paraphrase_scorer
import onmt_model
import numpy as np
import os
import spacy

nlp = spacy.load('en_core_web_sm')
from sears import replace_rules
import pickle
import time
import fasttext
import collections
from rule_picking import disqualify_rules
from rule_picking import choose_rules_coverage
import re

# Own imports
from transformers import BertForSequenceClassification, BertTokenizer
import torch

device = torch.device("cuda")
ps = paraphrase_scorer.ParaphraseScorer(gpu_id=0)


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
    pattern = '^##.*'
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


# test = ['[CLS]', 'terminal', '1', 'is', 'connected', 'to', 'the', 'negative', 'battery', 'terminal', '[SEP]', 'fjdaj',
#        '##f', '##test', 'is', 'it', 'tr', '##ue', '[SEP]']
# print(separate_answers(test))

# Tokenizers for Bert inputs and rules
tokenizer = replace_rules.Tokenizer(nlp)
pretrained_weights = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Own data
data_beetle = np.load('../data/sear_data/correct_beetle.npy', allow_pickle=True)

# Own model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load('../models/bert_asag/model.pt'))
model.cuda()
model.eval()

# data derived from correct beetle predictions, list of tuples of reference answer, student's answer and prediction
data = [(tuple([list_to_string(y) for y in separate_answers(x[0])]) + (x[1], )) for x in data_beetle]

def create_possible_flips(instance, model, topk=10, threshold=-10, ):
    """
    Finds possible flips given a data instance
    :param instance: tuple of list of bert tokens and label
    :param model: pytorch transformers model
    :param topk: max amount of paraphrases
    :param threshold:
    :return:
    """
    fs = []
    ref, stud, orig_pred = instance

    instance_for_onmt = onmt_model.clean_text(' '.join([x.text for x in nlp.tokenizer(stud)])
                                              , only_upper=False)
    paraphrases = ps.generate_paraphrases(instance_for_onmt, topk=topk, edit_distance_cutoff=4, threshold=threshold)

    texts = tokenizer.clean_for_model(tokenizer.clean_for_humans([x[0] for x in paraphrases]))
    # Let model classify the paraphrases and remember those, where it differs from the original prediction
    for i, sent in enumerate(texts):
        token_ids, segment, attention, lab = \
            BertPreprocessor(bert_tokenizer, data=[list_to_string(ref), list_to_string(sent), orig_pred]).load_data()
        token_ids = torch.tensor([token_ids]).long().to(device)
        segment = torch.tensor([segment]).long().to(device)
        attention = torch.tensor([attention]).long().to(device)
        outputs = model.forward(input_ids=token_ids, attention_mask=attention, token_type_ids=segment)
        label = np.argmax(outputs)
        if orig_pred != label:
            fs.append((sent, paraphrases[i][1]))

    return fs


orig_scores = {}
flips = collections.defaultdict(lambda: [])

# Find flips in data
for i, inst in enumerate(data):
    print("Data instance: ", i)
    fs = create_possible_flips(inst, model, topk=100, threshold=-10)
    # Key for the flips is the student's answer
    flips[inst[1]].extend([x[0] for x in fs])

#
tr2 = replace_rules.TextToReplaceRules(nlp, [x[1] for x in data], [], min_freq=0.005, min_flip=0.00,
                                       ngram_size=4)

# Finding frequent rules
frequent_rules = []
rule_idx = {}
rule_flips = {}
for z, f in enumerate(flips):
    # f is the student's answer
    # flips[f] flips for given student's answer
    rules = tr2.compute_rules(f, flips[f], use_pos=True, use_tags=True)
    for rs in rules:
        for r in rs:
            if r.hash() not in rule_idx:
                i = len(rule_idx)
                rule_idx[r.hash()] = i
                rule_flips[i] = []
                frequent_rules.append(r)
            i = rule_idx[r.hash()]
            rule_flips[i].append(z)
    print("Done with flip nr. ", z)

# Tokenize the student's answers
tokenized_stud_ans = tokenizer.tokenize([x[1] for x in data])
model_preds = {}
len(frequent_rules)

a = time.time()
rule_flips = {}
rule_other_texts = {}
rule_other_flips = {}
rule_applies = {}
for i, r in enumerate(frequent_rules):
    # Get indices, where rule can be applied
    idxs = list(tr2.get_rule_idxs(r))
    to_apply = [tokenized_stud_ans[x] for x in idxs]
    applies, nt = r.apply_to_texts(to_apply, fix_apostrophe=False)
    # Find indices, where rule has been applied
    applies = [idxs[x] for x in applies]
    # Find original data
    orig_data = [data[x] for x in applies]
    old_labels = [x[2] for x in orig_data]
    # Get indices
    to_compute = [x for x in zip(applies, nt) if x[1] not in model_preds]
    if to_compute:
        # New predicts
        new_labels = []
        # Very weird bug
        for compute in to_compute:
            j, new_stud = compute
            # Get reference answer for sequence classification
            orig_instance = data[j]
            token_ids, segment, attention, lab = \
                BertPreprocessor(bert_tokenizer,
                                 data=[orig_instance[0], new_stud, orig_instance[2]]).load_data()
            token_ids = torch.tensor([token_ids]).long().to(device)
            segment = torch.tensor([segment]).long().to(device)
            attention = torch.tensor([attention]).long().to(device)
            outputs = model.forward(input_ids=token_ids, attention_mask=attention, token_type_ids=segment)
            new_labels.append(np.argmax(outputs))
        for x, y in zip(to_compute, new_labels):
            model_preds[x[1]] = y
    # got error because of missing key
    new_labels = np.array([model_preds[x] for x in nt])
    where_flipped = np.where(new_labels != old_labels)[0]
    flips = sorted([applies[x] for x in where_flipped])
    rule_flips[i] = flips
    rule_other_texts[i] = nt
    rule_other_flips[i] = where_flipped
    rule_applies[i] = applies

print("Time used for applying rules: ", time.time() - a)

really_frequent_rules = [i for i in range(len(rule_flips)) if len(rule_flips[i]) > 1]
print("Amount of really frequent rules: ", len(really_frequent_rules))

# to_compute_score = collections.defaultdict(lambda: set())
# for i in really_frequent_rules:
#     orig_texts =  [right_val[z] for z in rule_applies[i]]
#     new_texts = rule_other_texts[i]
#     for o, n in zip(orig_texts, new_texts):
#         to_compute_score[o].add(n)

threshold = -7.15

orig_scores = {}
for i, t in enumerate(data):
    orig_scores[i] = ps.score_sentences(t[1], [t[1]])[0]

# I want rules s.t. the decile > -7.15. The current bottom 10% of a rule is always a lower bound on the decile, so if I see applies / 10 with score < -7.15 I can stop computing scores for that rule

ps_scores = {}

ps.last = None

rule_scores = []
rejected = set()
for idx, i in enumerate(really_frequent_rules):
    orig_texts = [data[z][1] for z in rule_applies[i]]
    orig_scor = [orig_scores[z] for z in rule_applies[i]]
    scores = np.ones(len(orig_texts)) * -50
    #     if idx in rejected:
    #         rule_scores.append(scores)
    #         continue
    decile = np.ceil(.1 * len(orig_texts))
    new_texts = rule_other_texts[i]
    bad_scores = 0
    for j, (o, n, orig) in enumerate(zip(orig_texts, new_texts, orig_scor)):
        if o not in ps_scores:
            ps_scores[o] = {}
        if n not in ps_scores[o]:
            if n == '':
                score = -40
            else:
                score = ps.score_sentences(o, [n])[0]
            ps_scores[o][n] = min(0, score - orig)
        scores[j] = ps_scores[o][n]
        if ps_scores[o][n] < threshold:
            bad_scores += 1
        if bad_scores >= decile:
            rejected.add(idx)
            break
    rule_scores.append(scores)

    print("Evaluated frequent rule nr. ", idx)

# import pickle
# pickle.dump({'ps_scores': ps_scores, 'orig_scores': orig_scores}, open('/home/marcotcr/tmp/polarity_scoresz.pkl', 'wb'))

print("Number of rules after rejection process: ", len(rule_scores) - len(rejected))

rule_flip_scores = [rule_scores[i][rule_other_flips[really_frequent_rules[i]]] for i in range(len(rule_scores))]

frequent_flips = [np.array(rule_applies[i])[rule_other_flips[i]] for i in really_frequent_rules]

rule_precsupports = [len(rule_applies[i]) for i in really_frequent_rules]

threshold = -7.15
# x = choose_rules_coverage(fake_scores, frequent_flips, frequent_supports,
disqualified = disqualify_rules(rule_scores, frequent_flips,
                                rule_precsupports,
                                min_precision=0.0, min_flips=6,
                                min_bad_score=threshold, max_bad_proportion=.10,
                                max_bad_sum=999999)

# [(i, x.hash()) for (i, x) in enumerate(frequent_rules) if 'text_movie -> text_film' in x.hash()]

threshold = -7.15
a = time.time()
x = choose_rules_coverage(rule_flip_scores, frequent_flips, None,
                          None, len(data),
                          frequent_scores_on_all=None, k=10, metric='max',
                          min_precision=0.0, min_flips=0, exp=True,
                          min_bad_score=threshold, max_bad_proportion=.1,
                          max_bad_sum=999999,
                          disqualified=disqualified,
                          start_from=[])
print("Rule coverage chosen. ", time.time() - a)
support_denominator = float(len(data))
soup = lambda x: len(rule_applies[really_frequent_rules[x]]) / support_denominator
prec = lambda x: frequent_flips[x].shape[0] / float(len(rule_scores[x]))
fl = len(set([a for r in x for a in frequent_flips[r]]))
print('Instances flipped: %d (%.2f)' % (fl, fl / float(len(data))))
print('\n'.join(['%-5d %-5d %-5d %-35s f:%d avg_s:%.2f bad_s:%.2f bad_sum:%d Prec:%.2f Supp:%.2f' % (
    i, x[i], really_frequent_rules[r],
    frequent_rules[really_frequent_rules[r]].hash().replace('text_', '').replace('pos_', '').replace('tag_', ''),
    frequent_flips[r].shape[0],
    np.exp(rule_flip_scores[r]).mean(), (rule_scores[r] < threshold).mean(),
    (rule_scores[r] < threshold).sum(), prec(r), soup(r)) for i, r in enumerate(x)]))

# ### a couple of examples from the first rules
for r in x:
    rid = really_frequent_rules[r]
    rule = frequent_rules[rid]
    print('Rule: %s' % rule.hash())
    print()
    for f in rule_flips[rid][:2]:
        print('%s\nP(positive):%.2f' % (data[f][1], model.predict_proba([data[f][1]])[0, 1]))
        print()
        new = rule.apply(tokenized_stud_ans[f])[1]
        print('%s\nP(positive):%.2f' % (new, model.predict_proba([new])[0, 1]))
        print()
        print()
    print('---------------')
