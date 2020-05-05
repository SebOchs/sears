import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append('sears')  # noqa
sys.path.append('../bachelor-thesis/bert')
from preprocessing_bert import BertPreprocessor
import paraphrase_scorer
import onmt_model
import numpy as np
import en_core_web_md

nlp = en_core_web_md.load()
from sears import replace_rules
import time
import collections
import re

# Own imports
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

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


# test = ['[CLS]', 'terminal', '1', 'is', 'connected', 'to', 'the', 'negative', 'battery', 'terminal', '[SEP]', 'fjdaj',
#        '##f', '##test', 'is', 'it', 'tr', '##ue', '[SEP]']
# print(separate_answers(test))

# Tokenizers for Bert inputs and rules
tokenizer = replace_rules.Tokenizer(nlp)
pretrained_weights = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Own data
val_data = np.load('../bachelor-thesis/models/bert_sciEntsBank/correct_sciEntsBank_val.npy', allow_pickle=True)

# Own model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load('../bachelor-thesis/models/bert_sciEntsBank/model.pt'))
model.cuda()
model.eval()

# data derived from correct scientsbank predictions, list of tuples of reference answer, student's answer and prediction
data = [separate_answers(x[0]) for x in val_data if x[1] == 0]


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
    ref, stud = instance
    ref = list_to_string(ref)
    stud = list_to_string(stud)
    instance_for_onmt = onmt_model.clean_text(' '.join([x.text for x in nlp.tokenizer(stud)])
                                              , only_upper=False)
    paraphrases = ps.generate_paraphrases(instance_for_onmt, topk=topk, edit_distance_cutoff=4, threshold=threshold)

    texts = tokenizer.clean_for_model(tokenizer.clean_for_humans([x[0] for x in paraphrases]))
    texts = [x.lower() for x in texts]
    # Let model classify the paraphrases and remember those, where it differs from the original prediction
    for i, sent in enumerate(texts):
        logits = predict(model, ref, sent, 0)
        if int(np.argmax(logits)) == 2:
            fs.append((sent, paraphrases[i][1]))

    return fs


orig_scores = {}
flips = collections.defaultdict(lambda: [])
# Find flips in data
for i, inst in enumerate(data):
    if i % 1 == 0:
        print("Data instance nr: ", i)
    fs = create_possible_flips(inst, model, topk=100, threshold=-10)
    # Key for the flips is the student's answer
    flips[list_to_string(inst[1])].extend([x[0] for x in fs])

tr2 = replace_rules.TextToReplaceRules(nlp, [list_to_string(x[1]) for x in data], [], min_freq=0.005, min_flip=0.00,
                                       ngram_size=4)
# Finding frequent rules
frequent_rules = []
rule_idx = {}
rule_flips = {}
for z, f in enumerate(flips):
    # f is the student's answer
    # flips[f] flips for given student's answer
    rules = tr2.compute_rules(f, flips[f], use_pos=True, use_tags=False)
    for rs in rules:
        for r in rs:
            if r.hash() not in rule_idx:
                i = len(rule_idx)
                rule_idx[r.hash()] = i
                rule_flips[i] = []
                frequent_rules.append(r)
            i = rule_idx[r.hash()]
            rule_flips[i].append(z)
    if z % 1 == 0:
        print("Done with flip nr. ", z)

# Tokenize the student's answers
tokenized_stud_ans = tokenizer.tokenize([list_to_string(x[1]) for x in data])
model_preds = {}
print("Number of frequent rules: ", len(frequent_rules))

a = time.time()
rule_flips = {}
rule_other_texts = {}
rule_other_flips = {}
rule_applies = {}
for i, r in enumerate(frequent_rules):
    if i % 100 == 0:
        print("Nr. of rules applied: ", i)
    # Get indices, where rule can be applied
    idxs = list(tr2.get_rule_idxs(r))
    to_apply = [tokenized_stud_ans[x] for x in idxs]
    applies, nt = r.apply_to_texts(to_apply, fix_apostrophe=False)
    # Find indices, where rule has been applied
    applies = [idxs[x] for x in applies]
    to_compute = [x for x in zip(applies, nt) if x[1] not in model_preds]
    if to_compute:
        # New predicts
        new_labels = []
        for compute in to_compute:
            j, new_stud = compute
            # Get reference answer for sequence classification
            orig_instance = data[j]
            logits = predict(model, list_to_string(orig_instance[0]), new_stud, 0)
            new_label = int(np.argmax(logits))
            new_labels.append(new_label)
        for x, y in zip(to_compute, new_labels):
            model_preds[x[1]] = y

    new_labels = np.array([model_preds[x] for x in nt])
    where_flipped = np.where(new_labels == 2)[0]
    flips_1 = sorted([applies[x] for x in where_flipped])
    rule_flips[i] = flips_1
    rule_other_texts[i] = nt
    rule_other_flips[i] = where_flipped
    rule_applies[i] = applies

print("Time used for applying rules: ", time.time() - a)
threshold = int(0.01*len(data))
really_frequent_rules_idx = [i for i in range(len(rule_flips)) if len(rule_flips[i]) > threshold]
print("Amount of really frequent rules: ", len(really_frequent_rules_idx))


high_number_rules = [frequent_rules[idx] for idx in really_frequent_rules_idx]
np.save("baseline_sears_rules.npy", high_number_rules)



