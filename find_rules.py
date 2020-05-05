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

# Tokenizers for Bert inputs and rules
tokenizer = replace_rules.Tokenizer(nlp)
pretrained_weights = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


# Used for Text Fooler
class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


# Text Fooler function #1
def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


# Text Fooler function #2
def text_fooler(text_ls, true_label, model, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
                import_score_threshold=-1., sim_score_threshold=0.7, sim_score_window=15, synonym_num=50,
                batch_size=32):
    adversaries = []
    # first check the prediction of the original text#
    ref_ans, stud_ans = text_ls
    stud_ans = list_to_string(stud_ans).split(" ")
    orig_logits = predict(model, ref_ans, stud_ans, true_label)
    orig_probs = F.softmax(orig_logits, dim=0)
    orig_label = torch.argmax(orig_probs).item()
    orig_prob = orig_probs.max().item()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(stud_ans)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(stud_ans)

        # get importance score
        leave_1_texts = [stud_ans[:ii] + ['[UNK]'] + stud_ans[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = []
        num_queries += len(leave_1_texts)

        for new_ans in leave_1_texts:
            new_logits = predict(model, ref_ans, new_ans, true_label)
            new_probs = F.softmax(new_logits, dim=0)
            leave_1_probs.append(new_probs)
        leave_1_probs = torch.stack(leave_1_probs)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and stud_ans[idx] not in stop_words_set:
                    words_perturb.append((idx, stud_ans[idx]))
            except:
                print(idx, len(stud_ans), import_scores.shape, stud_ans, len(leave_1_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)

        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = stud_ans[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = []
            new_labels = []
            for syn_text in new_texts:
                syn_logits = predict(model, ref_ans, syn_text, true_label)
                new_probs.append(F.softmax(syn_logits, dim=0))

            new_probs = torch.stack(new_probs)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window <= len_text - idx - 1:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window > len_text - idx - 1:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[
                    0]

            num_queries += len(new_texts)

            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (2 == torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos (maybe not)

            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]

            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            # Uncomment to inverse mask and only allow candidates where POS is not the same
            # pos_mask = np.invert(pos_mask)
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                adversaries.append(tuple(text_prime))
                break
            """
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
            adversaries.append(text_cache)
            #new_labels.append()
            """
        # Combine adversaries with new labels
        result = set(i for i in adversaries if i[0] != stud_ans[:])
        return num_changed, num_queries, result


# Own interpretation of Text Bugger
def text_bugger(text, true_label, model, max_amount=10):
    """
    Uses text-fooler importance scoring to find good candidates for modifications
    Uses TextBugger operations
    :param text: tuple of list of strings
    :param true_label: integer of [0, 1, 2]
    :param model: BERT Sequence Pair Classification Model with 3 Labels
    :return:
    """

    adjectives = list(np.load('final_adj.npy', allow_pickle=True))
    adverbs = list(np.load('final_adv.npy', allow_pickle=True))
    queries = {}
    successes = {}

    # Was used to track top adjectives/adverbs
    # track_words_adj = {}
    # track_words_adv = {}

    def number_attack(word, max_amount=2):
        """
        Converts characters in a string to numbers, returns permutation of possible letter-to-number modifications
        :param max_amount: Max. amount of replaced letters
        :param word: string
        :return: list of strings
        """
        candidates = []
        # dictionary of letters that will be replaced by the specific number
        num_chars = {"a": "4",
                     "e": "3",
                     "l": "1",
                     "o": "0",
                     "s": "5",
                     "t": "7"}
        possible_operations = []
        # Find all possible operations
        for idx, i in enumerate(word):
            if i in num_chars:
                possible_operations.append((idx, num_chars[i]))
        # Find all "1 to max amount" rank permutations
        while max_amount > 1:
            possible_operations.extend(list(itertools.permutations(possible_operations, max_amount)))
            max_amount = max_amount - 1
        # Apply operations to the word
        for combination in possible_operations:
            if type(combination[0]) == int:
                candidates.append(word[:combination[0]] + combination[1] + word[combination[0] + 1:])
            else:
                for modification in combination:
                    candidates.append(word[:modification[0]] + modification[1] + word[modification[0] + 1:])
        return list(set(candidates))

    def char_attack(word):
        """
        Creates all possible pairwise character switches
        :param word: string
        :return: list of strings
        """
        candidates = []
        for i in range(1, len(word)):
            if word[i - 1] != word[i]:
                if (i + 1) < len(word):
                    candidates.append(word[:i - 1] + word[i] + word[i - 1] + word[i + 1:])
                else:
                    candidates.append(word[:i - 1] + word[i] + word[i - 1])
        return list(set(candidates))

    def split_attack(word):
        candidates = []
        if 3 < len(word) < 7:
            for i in range(1, len(word) - 2):
                candidates.append(word[:i + 2] + ' ' + word[i + 2:])
        return list(set(candidates))

    def char_removal_attack(word):
        candidates = []
        if len(word) > 4:
            for i in range(1, len(word)):
                if word[i - 1] == word[i]:
                    candidates.append(word[:i - 1] + word[i:])
                if word[i] in 'aeiou':
                    if i == len(word) - 1:
                        candidates.append(word[:i])
                    else:
                        candidates.append(word[:i] + word[i + 1:])
        return list(set(candidates))

    def adverb_attack(word, adv=adverbs):
        candidates = []
        for i in adv:
            candidates.append([i, word])
        return candidates

    def adjective_attack(word, adj=adjectives):
        candidates = []
        for i in adj:
            candidates.append([i, word])
        return candidates

    adversaries = []
    # first check the prediction of the original text#
    ref_ans, stud_ans = text
    orig_logits = predict(model, ref_ans, stud_ans, true_label)
    orig_probs = F.softmax(orig_logits, dim=0)
    orig_label = torch.argmax(orig_probs).item()
    orig_prob = orig_probs.max().item()
    if true_label != orig_label:
        return 0, 0, []
    else:
        doc = nlp(" ".join(stud_ans))
        len_text = len(stud_ans)
        num_queries = 1

        # get importance score
        leave_1_texts = [stud_ans[:ii] + ['[UNK]'] + stud_ans[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = []
        num_queries += len(leave_1_texts)

        for new_ans in leave_1_texts:
            new_logits = predict(model, ref_ans, new_ans, true_label)
            new_probs = F.softmax(new_logits, dim=0)
            leave_1_probs.append(new_probs)
        leave_1_probs = torch.stack(leave_1_probs)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > -1:
                    words_perturb.append(idx)
            except:
                print(idx, len(stud_ans), import_scores.shape, stud_ans, len(leave_1_texts))

        # Use TextBugger operations to alter words
        for i in words_perturb:
            if stud_ans[i].isalpha():
                modified = {}
                tag = doc[i].pos_

                # Number Attack
                nmb_atk = number_attack(stud_ans[i])
                modified['number'] = nmb_atk
                queries['number'] = queries.get('number', 0) + len(nmb_atk)
                # Character Attack
                char_atk = char_attack(stud_ans[i])
                modified['char'] = char_atk
                queries['char'] = queries.get('char', 0) + len(char_atk)
                # Split Attack
                split_atk = split_attack(stud_ans[i])
                modified['split'] = split_atk
                queries['split'] = queries.get('split', 0) + len(split_atk)
                # Character Removal Attack
                char_removal_atk = char_removal_attack(stud_ans[i])
                modified['char_rem'] = char_removal_atk
                queries['char_rem'] = queries.get('char_rem', 0) + len(char_removal_atk)
                # Adverb and Adjective attack
                if tag == ('NOUN' or 'PRON' or 'PROPN'):
                    adj_atk = adjective_attack(stud_ans[i])
                    modified['adj'] = adj_atk
                    queries['adj'] = queries.get('adj', 0) + len(adj_atk)
                if tag == 'VERB':
                    adv_atk = adverb_attack(stud_ans[i])
                    modified['adv'] = adv_atk
                    queries['adv'] = queries.get('adv', 0) + len(adv_atk)

                for j in modified.keys():
                    for x in modified[j]:
                        modified_ans = stud_ans.copy()
                        if type(x) == list:
                            modified_ans[i:i + 1] = x
                        else:
                            modified_ans[i] = x
                        modified_logits = predict(model, ref_ans, modified_ans, 0)
                        modified_label = torch.argmax(modified_logits).item()
                        if modified_label == 2:
                            successes[j] = successes.get(j, 0) + 1
                            adversaries.append(modified_ans)
                            # print(list_to_string(ref_ans), orig_prob, list_to_string(stud_ans), F.softmax(modified_logits, dim=0).max().item(), list_to_string(modified_ans))
                            # Was used to track top adjectives and adversaries
                            """
                            if tag == ('NOUN' or 'PRON' or 'PROPN'):
                                track_words_adj[x[0]] = track_words_adj.get(x[0], 0) + 1
                            if tag == 'VERB':
                                track_words_adv[x[0]] = track_words_adv.get(x[0], 0) + 1
                            """

    return queries, successes, adversaries,  # track_words_adj, track_words_adv


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


def main():
    # Own data
    val_data = np.load('../bachelor-thesis/models/bert_scientsBank/correct_sciEntsBank_val.npy', allow_pickle=True)
    # Own model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.load_state_dict(torch.load('../bachelor-thesis/models/bert_sciEntsBank/model.pt'))
    model.cuda()
    model.eval()
    # data derived from correct model predictions, list of tuples of reference answer, student's answer and prediction
    # All cases are incorrect
    data = [separate_answers(x[0]) for x in val_data if x[1] == 0]

    # TextFooler part
    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}
    stop_words_set = criteria.get_stopwords()
    print("Building vocab...")
    with open("../TextFooler/data/counter-fitted-vectors.txt", 'r', encoding="utf8") as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    cos_sim = np.load("../TextFooler/data/cos_sim_counter_fitting.npy", allow_pickle=True)
    print("Cos sim import finished!")
    use = USE("use")
    print('Start attacking!')
    orig_scores = {}
    flips = collections.defaultdict(lambda: [])
    # Find flips in data
    adversary_successes = {}
    adversary_count = {}
    # Was used to track top adjectives/adverbs
    # main_tracker_adv = {}
    # main_tracker_adj = {}
    for i, inst in enumerate(data):
        print("Data instances finished: ", i)
        adversaries = []
        # Baseline run:
        num_tf_changed, num_tf_queries, tf_adversaries = text_fooler(inst, 0, model, stop_words_set,
                                                                     word2idx, idx2word, cos_sim, sim_predictor=use,
                                                                     sim_score_threshold=0.7,
                                                                     import_score_threshold=-1.,
                                                                     sim_score_window=4,
                                                                     synonym_num=50,
                                                                     batch_size=16)
        # Uncomment for textfooler only
        # query_num, success_num, bug_adversaries = text_bugger(inst, 0, model)
        # Was used to track top adjectives and adversaries
        """, tracker_adj, tracker_adv"""

        # All adversaries
        adversaries.extend(tf_adversaries)
        #adversaries.extend(bug_adversaries)

        # Was used to track top adjectives and adversaries
        """
        for key in tracker_adj:
            main_tracker_adj[key] = main_tracker_adj.get(key, 0) + tracker_adj[key]
        for key in tracker_adv:
            main_tracker_adv[key] = main_tracker_adv.get(key, 0) + tracker_adv[key]
        """

        if len(adversaries) > 0:
            flips[list_to_string(inst[1])].extend(adversaries)
            adversary_successes['tf'] = adversary_successes.get('tf', 0) + num_tf_changed
            adversary_count['tf'] = adversary_count.get('tf', 0) + num_tf_queries
        #    for key in query_num:
        #        adversary_successes[key] = adversary_successes.get(key, 0) + success_num.get(key, 0)
        #        adversary_count[key] = adversary_count.get(key, 0) + query_num.get(key, 0)

    # Was used to track top adjectives and adversaries
    # np.save("adv_result.npy", main_tracker_adv)
    # np.save("adj_result.npy", main_tracker_adj)
    np.save("bs_adversary_successes_tf.npy", adversary_successes)
    np.save("bs_adversary_count_tf.npy", adversary_count)
    tr2 = replace_rules.TextToReplaceRules(nlp, [list_to_string(x[1]) for x in data], [], min_freq=0.005, min_flip=0.005,
                                           ngram_size=2)

    # Finding frequent rules
    frequent_rules = []
    rule_idx = {}
    rule_flips = {}
    for z, f in enumerate(flips):
        # f is the student's answer
        # flips[f] flips for given student's answer
        rules = tr2.compute_rules(f, [list_to_string(x) for x in flips[f]], use_pos=True, use_tags=False)
        for rs in rules:
            for r in rs:
                if r.hash() not in rule_idx:
                    i = len(rule_idx)
                    rule_idx[r.hash()] = i
                    rule_flips[i] = []
                    frequent_rules.append(r)
                i = rule_idx[r.hash()]
                rule_flips[i].append(z)
        if z % 1000 == 0:
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
                logits = predict(model, orig_instance[0], new_stud, 0)
                new_label = int(np.argmax(logits))
                new_labels.append(new_label)
            for x, y in zip(to_compute, new_labels):
                model_preds[x[1]] = y

        new_labels = np.array([model_preds[x] for x in nt])
        where_flipped = np.where(new_labels == 2)[0]
        flips = sorted([applies[x] for x in where_flipped])
        rule_flips[i] = flips
        rule_other_texts[i] = nt
        rule_other_flips[i] = where_flipped
        rule_applies[i] = applies

    print("Time used for applying rules: ", time.time() - a)

    threshold = int(0.01*len(data))
    really_frequent_rules_idx = [i for i in range(len(rule_flips)) if len(rule_flips[i]) > threshold]



    # test = [frequent_rules[i] for i in really_frequent_rules_idx if frequent_rules[i].hash().split()[1] == '->']
    # test_2 = [i.hash() for i in test if i.hash()[:4] == 'text']
    print("Amount of really frequent rules: ", len(really_frequent_rules_idx))

    print("Done!")
    high_number_rules = [frequent_rules[idx] for idx in really_frequent_rules_idx]
    np.save("bs_tf_rules.npy", high_number_rules)


if __name__ == "__main__":
    main()
