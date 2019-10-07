# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np
from embedding import PreTrainedEmbeddings

class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs
        self.scorer = ProbabilisticSequenceScorer(tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs)

    def viterbi(self, sentence_tokens: List[Token], pred_tags):
        # generate a viterbi score matrix whose size is number of states * steps
        num_tags = len(self.tag_indexer)
        score = np.zeros((len(sentence_tokens), num_tags))
        backpointers = np.zeros((len(sentence_tokens) - 1, num_tags))

        # initialize start probability
        score[0, :] = [self.scorer.score_init(sentence_tokens, tag_idx)
                       + self.scorer.score_emission(sentence_tokens, tag_idx, 0) for tag_idx in range(num_tags)]

        # build up the score matrix
        for step in range(1, len(sentence_tokens)):
            for tag in range(num_tags):
                transition_prob = [self.scorer.score_transition(sentence_tokens, prev_tag, tag)
                                   for prev_tag in range(num_tags)]
                emission_prob = self.scorer.score_emission(sentence_tokens, tag, step)
                probability = score[step - 1] + transition_prob + emission_prob
                backpointers[step - 1, tag] = np.argmax(probability)
                score[step, tag] = np.max(probability)

        # backtrack to generate most likely sequence
        sequence = np.zeros(len(sentence_tokens))
        last_tag_idx = np.argmax(score[len(sentence_tokens) - 1, :])
        sequence[0] = last_tag_idx
        backtrack_idx = 1
        for i in range(len(sentence_tokens) - 2, -1, -1):
            sequence[backtrack_idx] = backpointers[i, int(last_tag_idx)]
            last_tag_idx = backpointers[i, int(last_tag_idx)]
            backtrack_idx += 1
        sequence = np.flip(sequence, axis=0)

        # translate to bio tags:
        for tag_idx in sequence:
            pred_tags.append(self.tag_indexer.get_object(tag_idx))


    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        pred_tags = []
        self.viterbi(sentence_tokens, pred_tags)
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)

class FeatureBasedSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, feature_indexer: Indexer, weights, wordvec_index):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.weights = weights
        self.wordvec_index = wordvec_index

    def _transition_feature(self, prev_tag_idx: int, curr_tag_idx: int):
        """return transition probability either 1 or 0 based on hard constraints"""
        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
        if prev_tag_idx == -1 and curr_tag.startswith('I'):
            return 0
        elif prev_tag_idx == -1:
            return 0.3

        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
        if prev_tag.startswith('O') and curr_tag.startswith('I') \
            or prev_tag.startswith('B') and curr_tag.startswith('I') and prev_tag[-3:] != curr_tag[-3:]:
            return 0
        return 0.5

    def score_transition(self, prev_tag_idx: int, curr_tag_idx: int):
        return self._transition_feature(prev_tag_idx, curr_tag_idx)

    def score_potential(self, sparse_feature):
        return np.sum(self.weights[sparse_feature])

    def score_wordvec(self, curr_tag_idx, word_vecs):
        return np.dot(self.weights[self.wordvec_index[curr_tag_idx]], word_vecs)

class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, embeddings):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        # build wordvec inder
        num_tags = len(self.tag_indexer)
        embeddings_dim = 100
        self.wordvec_index = np.array([np.arange(len(self.feature_indexer) + embeddings_dim * (i - num_tags),
                                                 len(self.feature_indexer) + embeddings_dim * (i + 1 - num_tags))
                              for i in range(num_tags)])
        self.scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_indexer, self.feature_weights, self.wordvec_index)
        self.embeddings = embeddings

    def _forward_backward(self, features, word_vecs):
        # generate a viterbi score matrix whose size is number of states * steps
        num_tags = len(self.tag_indexer)

        # initialize start probability
        self.alpha = np.zeros((len(features), num_tags))
        self.alpha[0, :] = [self.scorer.score_potential(features[0][tag_idx])
                            + self.scorer.score_wordvec(tag_idx, word_vecs[0]) for tag_idx in range(num_tags)]

        # build up the forward matrix
        for step in range(1, len(features)):
            for tag in range(num_tags):
                for prev_tag in range(num_tags):
                    self.alpha[step, tag] = np.logaddexp(self.alpha[step, tag], self.alpha[step - 1, prev_tag])
                self.alpha[step, tag] += self.scorer.score_potential(features[step][tag]) + self.scorer.score_wordvec(tag, word_vecs[step])

        # initialize backward probablity
        self.beta = np.zeros((len(features), num_tags))
        self.beta[len(features) - 1, :] = np.zeros(num_tags)

        # build up the backward matrix
        for step in range(len(features) - 2, -1, -1):
            for tag in range(num_tags):
                for next_tag in range(num_tags):
                    self.beta[step, tag] = np.logaddexp(self.beta[step, tag],
                                                        self.beta[step + 1, next_tag] \
                                                        + self.scorer.score_potential(features[step + 1][next_tag])
                                                        + self.scorer.score_wordvec(next_tag, word_vecs[step + 1])
                                                        + self.scorer.score_transition(tag, next_tag))

        # calculate normalization factor
        foward_backward = np.exp(self.alpha + self.beta)
        Z = np.sum(foward_backward, axis=1)
        self.marginal = np.array([foward_backward[i] / Z[i] for i in range(len(foward_backward))])

    def fit(self, features, gold_labels, word_vecs):
        num_tags = len(self.tag_indexer)
        self._forward_backward(features, word_vecs)
        for i in range(len(features)):
            # accumulate gold features
            for feat in features[i][self.tag_indexer.index_of(gold_labels[i])]:
                self.gradient[feat] += 1
            j = 0
            for feat in self.wordvec_index[self.tag_indexer.index_of(gold_labels[i])]:
                self.gradient[feat] += word_vecs[i][j]
                j += 1
            # substract marginals
            for tag in range(num_tags):
                marginals = self.marginal[i, tag]
                for feat in features[i][tag]:
                    self.gradient[feat] -= marginals
                j = 0
                for feat in self.wordvec_index[tag]:
                    self.gradient[feat] -= marginals * word_vecs[i][j]
                    j += 1

    def zero_gradients(self):
        self.gradient = Counter()

    def get_gradients(self):
        return self.gradient

    def viterbi(self, features, word_vecs):
        # generate a viterbi score matrix whose size is number of states * steps
        num_tags = len(self.tag_indexer)
        score = np.zeros((len(features), num_tags))
        backpointers = np.zeros((len(features) - 1, num_tags))

        # initialize start probability
        score[0, :] = [self.scorer.score_potential(features[0][tag_idx]) + self.scorer.score_wordvec(tag_idx, word_vecs[0]) for tag_idx in range(num_tags)]

        # build up the score matrix
        for step in range(1, len(features)):
            for tag in range(num_tags):
                potential = [ self.scorer.score_transition(prev_tag, tag) +
                             score[step - 1, prev_tag] for prev_tag in range(num_tags)]
                score[step, tag] = np.max(potential) + self.scorer.score_potential(features[step][tag]) + self.scorer.score_wordvec(tag, word_vecs[step])
                backpointers[step - 1, tag] = np.argmax(potential)

        # backtrack to generate most likely sequence
        sequence = np.zeros(len(features))
        last_tag_idx = np.argmax(score[len(features) - 1, :])
        sequence[0] = last_tag_idx
        backtrack_idx = 1
        for i in range(len(features) - 2, -1, -1):
            sequence[backtrack_idx] = backpointers[i, int(last_tag_idx)]
            last_tag_idx = backpointers[i, int(last_tag_idx)]
            backtrack_idx += 1
        sequence = np.flip(sequence, axis=0)

        # translate to bio tags:
        pred_tags = []
        for tag_idx in sequence:
            pred_tags.append(self.tag_indexer.get_object(tag_idx))
        return pred_tags

    def decode(self, sentence_tokens):
        features = [[[] for k in range(len(self.tag_indexer))] for i in range(len(sentence_tokens))]
        word_vecs = [[] for j in range(len(sentence_tokens))]
        for word_idx in range(len(sentence_tokens)):
            word_vecs[word_idx] = extract_embedding_features(sentence_tokens, word_idx, self.embeddings)
            for tag_idx in range(len(self.tag_indexer)):
                features[word_idx][tag_idx] = extract_emission_features(sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)
        pred_tags = self.viterbi(features, word_vecs)
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    true_labels = []
    for sentence in sentences:
        labels = []
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
            labels.append(tag)
        true_labels.append(labels)
    print("Extracting features")
    feature_indexer = Indexer()
    print("Initializing Word Vector")
    embeddings = PreTrainedEmbeddings.from_embeddings_file('data/glove/glove.6B.100d.txt')
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    word_vecs = [[[] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            word_vecs[sentence_idx][word_idx] = extract_embedding_features(sentences[sentence_idx].tokens, word_idx, embeddings)
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens,
                                                                                           word_idx, tag_indexer.get_object(tag_idx),
                                                                                           feature_indexer, add_to_indexer=True)
    # add word vector feature index
    for tag_idx in range(len(tag_indexer)):
        for i in range(len(word_vecs[0][0])):
            maybe_add_feature([], feature_indexer, True, tag_indexer.get_object(tag_idx)+ ":WordVec=" + str(i))
    print("Training")
    # initialize required parameters
    weights = np.zeros(len(feature_indexer))
    crf = CrfNerModel(tag_indexer, feature_indexer, weights, embeddings)
    optimizer = UnregularizedAdagradTrainer(weights, eta=0.055)

    # train model
    num_epoch = 10
    batch_size = len(feature_cache) // num_epoch
    for epoch_index in range(num_epoch):
        print("Training epoch: (%d/10)" % (epoch_index + 1))
        sentence_idx = batch_size * epoch_index
        if epoch_index == num_epoch - 1:
            end_idx = len(feature_cache)
        else:
            end_idx = batch_size * epoch_index + batch_size
        while sentence_idx < end_idx:
                crf.zero_gradients()
                crf.fit(feature_cache[sentence_idx], true_labels[sentence_idx], word_vecs[sentence_idx])
                optimizer.apply_gradient_update(crf.get_gradients(), 1)
                sentence_idx += 1

    # for i in range(100):
    #     crf.zero_gradients()
    #     crf.fit(feature_cache[i], true_labels[i], word_vecs[i])
    #     optimizer.apply_gradient_update(crf.get_gradients(), 1)
    return crf

def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param embeddings: embeddings class that stores word embeddings
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))

    return np.asarray(feats, dtype=int)

def extract_embedding_features(sentence_tokens: List[Token], word_index: int, embeddings):
    curr_word = sentence_tokens[word_index].word
    return embeddings.get_embedding(curr_word.lower())