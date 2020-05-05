import numpy as np
import sys
from nltk.corpus import stopwords
import collections
import click

# Extern imports
sys.path.append('sears')  # noqa
from sears import replace_rules

stop_words = stopwords.words('english')

def main():
    frequent_rules = np.load("bs_sears_final_rules.npy", allow_pickle=True)
    # Filter rules
    rejected_rules = []

    for i in frequent_rules:
        op_seq = i.op_sequence
        rep_seq = i.replace_sequence


        # If the replace sequence only contains text tokens and the op sequence only pos tokens, the rule is too general
        if (all((x.type == 'pos') for x in op_seq)) and (all(y.type == 'text' for y in rep_seq)):
            rejected_rules.append(i)

        # If the replace sequence replaces any pos with text, the rule is not viable
        pos_tags = ['NOUN', 'VERB', 'PRON', 'ADJ', 'ADV', 'ADP', 'DET']
        for tag in pos_tags:
            if any((x.type == 'pos' and x.value == tag) for x in op_seq) \
                    and not any((x.type == 'pos' and x.value == tag) for x in rep_seq):
                rejected_rules.append(i)

        # Sequences of verbs where one of them gets replaced by text are not viable
        if len([op_seq[i] for i in range(len(op_seq)) if (op_seq[i].type == 'pos' and op_seq[i].value == 'VERB')]) > \
                len([rep_seq[i] for i in range(len(rep_seq)) if
                     (rep_seq[i].type == 'pos' and rep_seq[i].value == 'VERB')]):
            rejected_rules.append(i)
        # As adjectives and adverbs get inserted before nouns / verbs, everything else is unnecessary
        if any((x.type == 'pos') for x in op_seq) and any((x.type == 'pos') for x in rep_seq):
            # create bigrams from replacing sequence
            bigrams = list(zip(rep_seq, rep_seq[1:]))
            for bigram in bigrams:
                if bigram[0].type == 'pos' and bigram[1].type == 'text':
                    rejected_rules.append(i)


    almost_final_rules = list(set(frequent_rules) - set(rejected_rules))
    # Manual review
    good_rules = [0, 1, 2, 3, 6, 14, 15, 17, 18, 19, 21, 31, 33, 34, 36, 37, 38, 43]
    final_rules = [almost_final_rules[idx] for idx in good_rules]
    np.save('bs_sears_final_rules.npy', final_rules)
    print('Amount of final rules: ', len(final_rules))


if __name__ == '__main__':
    main()
