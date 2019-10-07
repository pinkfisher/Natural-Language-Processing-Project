# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List

organizations = set()
locations = set()
names = set()
misc = set()

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


class PersonExample(object):
    """
    Data wrapper for a single sentence for person classification, which consists of many individual tokens to classify.

    Attributes:
        tokens: the sentence to classify
        labels: 0 if non-person name, 1 if person name for each token in the sentence
    """
    def __init__(self, tokens: List[str], labels: List[int]):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)


def transform_for_classification(ner_exs: List[LabeledSentence]):
    """
    :param ner_exs: List of chunk-style NER examples
    :return: A list of PersonExamples extracted from the NER data
    """
    global organizations, locations, no_chunks

    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        names.update([labeled_sent.tokens[i].word for i in range(len(tags)) if tags[i].endswith("PER")])
        organizations.update([labeled_sent.tokens[i].word for i in range(len(tags)) if tags[i].endswith("ORG")])
        locations.update([labeled_sent.tokens[i].word for i in range(len(tags)) if tags[i].endswith("LOC")])
        misc.update([labeled_sent.tokens[i].word for i in range(len(tags)) if tags[i].endswith("O") and labeled_sent.tokens[i].word.istitle()])
        yield PersonExample([tok.word for tok in labeled_sent.tokens], labels)


class CountBasedPersonClassifier(object):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """
    def __init__(self, pos_counts: Counter, neg_counts: Counter):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens: List[str], idx: int):
        if self.pos_counts[tokens[idx]] > self.neg_counts[tokens[idx]]:
            return 1
        else:
            return 0


def train_count_based_binary_classifier(ner_exs: List[PersonExample]):
    """
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedPersonClassifier using counts collected from the given examples
    """
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts[ex.tokens[idx]] += 1.0
            else:
                neg_counts[ex.tokens[idx]] += 1.0
    print(repr(pos_counts))
    print(repr(pos_counts["Peter"]))
    print(repr(pos_counts["aslkdjtalk;sdjtakl"]))
    return CountBasedPersonClassifier(pos_counts, neg_counts)


class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, indexer: Indexer):
        self.weights = weights
        self.indexer = indexer

    def __sigmoid(self, score):
        """
        Generate the probability of being positive using logistic function
        :param score:
        :return: probability of being positive
        """
        return 1 / (1 + np.exp(-score))

    def fit(self, tokens: List[str], idx: int, label: int):
        """
        Train for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :param label: true label
        """
        sparse_features = np.zeros(len(self.weights))
        features = Counter()
        features["IsTitle"] = tokens[idx].istitle()
        if idx > 0:
            features["IsPrevTitle"] = tokens[idx - 1].istitle()
        features["IsUpper"] = tokens[idx].isupper()
        features["IsOrganization"] = tokens[idx] in organizations
        features["IsLocation"] = tokens[idx] in locations
        features["IsName"] = tokens[idx] in names
        features["IsMisc"] = tokens[idx] in misc
        features["Bias"] = 1
        for key in features.keys():
            sparse_features[self.indexer.index_of(key)] = features[key]

        likelihood = self.__sigmoid(sum(sparse_features * self.weights))
        self.gradients = Counter()
        for key in features.keys():
            self.gradients[self.indexer.index_of(key)] = features[key] * (label - likelihood)

    def get_gradients(self):
        """
        :return: Return the gradient of this training sample
        """
        return self.gradients

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        sparse_features = np.zeros(len(self.weights))
        features = Counter()
        features["IsTitle"] = tokens[idx].istitle()
        if idx > 0:
            features["IsPrevTitle"] = tokens[idx - 1].istitle()
        features["IsUpper"] = tokens[idx].isupper()
        features["IsOrganization"] = tokens[idx] in organizations
        features["IsLocation"] = tokens[idx] in locations
        features["IsName"] = tokens[idx] in names
        features["IsMisc"] = tokens[idx] in misc
        features["Bias"] = 1
        for key in features.keys():
            sparse_features[self.indexer.index_of(key)] = features[key]

        likelihood = self.__sigmoid(np.sum(sparse_features * self.weights))
        if likelihood >= 0.5:
            return 1
        else:
            return 0

def train_classifier(ner_exs: List[PersonExample]):
    """
    :param ner_exs: training examples
    :return: A PersonClassifier using indicator features
    """

    # extract features
    vocab = Indexer()
    vocab.add_and_get_index("IsUpper")
    vocab.add_and_get_index("IsTitle")
    vocab.add_and_get_index("IsLocation")
    vocab.add_and_get_index("IsOrganization")
    vocab.add_and_get_index("IsMisc")
    vocab.add_and_get_index("IsName")
    vocab.add_and_get_index("Bias")

    # one more feature for bias
    weights = np.zeros(len(vocab) + 1)
    classifier = PersonClassifier(weights, vocab)
    optimizer = SGDOptimizer(weights, 0.1)

    # train model
    num_epoch = 10
    batch_size = len(ner_exs) // num_epoch
    for epoch_index in range(num_epoch):
        print("Training epoch: (%d/10)" % (epoch_index + 1))
        if epoch_index == num_epoch - 1:
            samples = ner_exs[batch_size * epoch_index:]
        else:
            samples = ner_exs[batch_size * epoch_index : batch_size * epoch_index + batch_size]
        for s in samples:
            for idx in range(0, len(s)):
                classifier.fit(s.tokens, idx, s.labels[idx])
                optimizer.apply_gradient_update(classifier.get_gradients(), 1)

    return classifier




def evaluate_classifier(exs: List[PersonExample], classifier: PersonClassifier):
    """
    Prints evaluation of the classifier on the given examples
    :param exs: PersonExample instances to run on
    :param classifier: classifier to evaluate
    """
    predictions = []
    golds = []
    for ex in exs:
        for idx in range(0, len(ex)):
            golds.append(ex.labels[idx])
            predictions.append(classifier.predict(ex.tokens, idx))
    print_evaluation(golds, predictions)


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


def predict_write_output_to_file(exs: List[PersonExample], classifier: PersonClassifier, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))
    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
    else:
        classifier = train_classifier(train_class_exs)
    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train accuracy===")
    evaluate_classifier(train_class_exs, classifier)
    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs, classifier)
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))



