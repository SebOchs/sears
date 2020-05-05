import nltk
import sys

sys.path.append('../TextFooler')
import criteria
import numpy as np


def find_top_k_words_with_tag(k, tag):
    stop_words = criteria.get_stopwords()
    bigrams = nltk.bigrams((x[0].lower(), x[1]) for x in nltk.corpus.brown.tagged_words(tagset='universal'))
    # Filter bigrams
    tagged = []
    if tag == 'ADJ':
        next_tags = 'PROPN', 'NOUN', 'PRON'
        tagged = [x[0] for x in bigrams if x[0][1] == 'ADJ' and x[1][1] in next_tags]

    if tag == 'ADV':
        next_tags = 'VERB'
        tagged = [x[0] for x in bigrams if x[0][1] == 'ADV' and x[1][1] in next_tags]
        # tagged.extend([x[1] for x in bigrams if x[1][1] == 'ADV' and x[0][1] in next_tags])
    freq = nltk.FreqDist(x for x in tagged if x[0] not in stop_words) \
        .most_common(k)
    top_list = [x[0][0] for x in freq]

    return top_list


def best_words_percentile(words, percentage=0.7):
    words = sorted(words.items(), key=lambda item: item[1], reverse=True)
    overall_sum = np.sum([x[1] for x in words])
    best = []
    iter_sum = 0
    for i in range(len(words)):
        if iter_sum <= overall_sum * percentage:
            best.append(words[i])
            iter_sum += words[i][1]
        else:
            break
    return [x[0] for x in best]


def main():
    """
    top_adjectives = find_top_k_words_with_tag(100, 'ADJ')
    top_adverbs = find_top_k_words_with_tag(100, 'ADV')

    dict = {}
    dict['ADJ'] = top_adjectives
    dict['ADV'] = top_adverbs
    np.save("top_adjectives_adverbs.npy", dict)
    """
    adj_res = np.load('adj_result.npy', allow_pickle=True).item()
    adv_res = np.load('adv_result.npy', allow_pickle=True).item()
    best_adjectives = best_words_percentile(adj_res)
    best_advberbs = best_words_percentile(adv_res)
    np.save('final_adj.npy', best_adjectives)
    np.save('final_adv.npy', best_advberbs)


if __name__ == "__main__":
    main()
