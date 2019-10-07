# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from sentiment_data import *
from typing import List
from argparse import Namespace

from utils import Indexer


def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

class ReviewDataset(Dataset):
    def __init__(self, data_bundle, vectorizer):
        """:param data_bundle: a dict in the format of {'train': (indexed words, sentence length, labels),
                                                        'dev': (indexed words, sentence length, labels),
                                                        'test': (indexed words, sentence length, labels)"""
        self.train_set, self.train_len, self.train_labels = data_bundle['train']
        self.train_size = len(self.train_set)

        self.dev_set, self.dev_len, self.dev_labels = data_bundle['dev']
        self.dev_size = len(self.dev_set)

        self._lookup_dict = {'train': (self.train_set, self.train_len, self.train_size, self.train_labels),
                             'dev': (self.dev_set, self.dev_len, self.dev_size, self.dev_labels)}
        self._vectorizer = vectorizer

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, data_bundle, word_vectors, unique_labels, Vectorizer):
        return cls(data_bundle, Vectorizer.from_word_vectors(word_vectors, unique_labels))

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split='train'):
        self._target_split = split
        self._target_set, self._target_len, self._target_size, self._target_labels = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_set[index]
        len = self._target_len[index]
        label = self._target_labels[index]

        review_vector = self._vectorizer.vectorize(row)

        rating_index = self._vectorizer.rating_vocab.index_of(label)

        return {'x_data': review_vector, 'x_len': len, 'y_target': rating_index}


class ReviewVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, review_vocab: WordEmbeddings, rating_vocab):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """Create a collapsed one-hit vector for the review"""
        embeddings = np.zeros(self.review_vocab.get_embedding_length(), dtype=np.float32)

        for word_index in review:
            embeddings += self.review_vocab.vectors[word_index]

        return embeddings / len(review)

    @classmethod
    def from_word_vectors(cls, word_vectors, unique_labels):
        """Instantiate the vectorizer"""
        review_vocab = word_vectors
        rating_vocab = Indexer()

        # Add ratings
        for l in unique_labels:
            rating_vocab.add_and_get_index(l)

        return cls(review_vocab, rating_vocab)


class ReviewSequenceVectorizer(ReviewVectorizer):

    def __init__(self, review_vocab: WordEmbeddings, rating_vocab):
        super(ReviewSequenceVectorizer, self).__init__(review_vocab, rating_vocab)

    def vectorize(self, review):
        """Create a sequence word vector"""
        embeddings = []

        for word_index in review:
            embeddings.append(self.review_vocab.vectors[word_index])

        return np.asarray(embeddings)

    @classmethod
    def from_word_vectors(cls, word_vectors, unique_labels):
        """Instantiate the vectorizer"""
        review_vocab = word_vectors
        rating_vocab = Indexer()

        # Add ratings
        for l in unique_labels:
            rating_vocab.add_and_get_index(l)

        return cls(review_vocab, rating_vocab)


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """Predict the rating of a review"""
    vectorized_review = torch.tensor(vectorizer.vectorize(review)).unsqueeze(0)
    result = classifier(vectorized_review.float())

    probability_value = torch.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.index_of(index)


class FFNN(nn.Module):
    def __init__(self, inp, hid, out):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.fc2 = nn.Linear(hid, out)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        intermediate_vector = F.relu(self.fc1(x))
        prediction_vector = self.fc2(F.dropout(intermediate_vector, p=0.5)).squeeze()
        return prediction_vector


# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample],
                        dev_exs: List[SentimentExample], test_exs: List[SentimentExample],
                        word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs], dtype=int)
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs], dtype=int)
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])
    data_bundle = {'train': (train_mat, train_seq_lens, train_labels_arr),
                   'dev': (dev_mat, dev_seq_lens, dev_labels_arr)}
    unique_labels = np.unique(np.concatenate([train_labels_arr, dev_labels_arr]))

    # define NN parameters
    args = Namespace(
        input_size=len(word_vectors.vectors[0]),
        hidden_size=300,
        output_size=1,
        num_epoch=20,
        batch_size=128,
        learning_rate=0.001,
        seed=1337,
        device='cpu',
        catch_keyboard_interrupt=True,
    )

    dataset = ReviewDataset.load_dataset_and_make_vectorizer(data_bundle, word_vectors, unique_labels, ReviewVectorizer)
    vectorizer = dataset.get_vectorizer()
    classifier = FFNN(args.input_size, args.hidden_size, args.output_size)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    loss_func = nn.BCEWithLogitsLoss()

    # train
    for epoch_index in range(args.num_epoch):
        print("Epoch: ({}/{})".format(epoch_index + 1, args.num_epoch))
        dataset.set_split('train')

        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()

            y_pred = classifier(x=batch_dict['x_data'].float())

            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            loss.backward()

            optimizer.step()

            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        print('Training Accuracy', running_acc)

        # test on development set
        dataset.set_split('dev')
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0
        running_acc = 0
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = classifier(x=batch_dict['x_data'].float())

            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            loss.backward()

            optimizer.step()

            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        print('Dev Accuracy', running_acc)


    # predict on test set
    results = []
    for line in test_exs:
        padded_line = np.asarray(pad_to_length(np.array(line.indexed_words), seq_max_len), dtype=int)
        line.label = predict_rating(padded_line, classifier, vectorizer)
        results.append(line)
    return results


class RNN(nn.Module):
    def __init__(self, inp, hid, out, n_layers,
                 batch_first=True, bidirectional=True, dropout_p=0.5, padding_idx=0):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(inp, hid, num_layers=n_layers, bidirectional=bidirectional,
                          dropout=dropout_p, batch_first=batch_first)
        self.fc = nn.Linear(hid * 2, out)
        self._dropout_p = dropout_p

    def forward(self, x):
        # output = [batch size, sent len, hid dim * num directions]
        y_out, (hidden, cell) = self.rnn(x)
        y_out = self.fc(F.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1), p=self._dropout_p)).squeeze()

        return y_out



# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a Bi-LSTM neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input. The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs], dtype=int)
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs], dtype=int)
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])
    data_bundle = {'train': (train_mat, train_seq_lens, train_labels_arr),
                   'dev': (dev_mat, dev_seq_lens, dev_labels_arr)}
    unique_labels = np.unique(np.concatenate([train_labels_arr, dev_labels_arr]))

    # define NN parameters
    args = Namespace(
        input_size=len(word_vectors.vectors[0]),
        hidden_size=100,
        output_size=1,
        num_layers=2,
        num_epoch=20,
        batch_size=128,
        learning_rate=0.01,
        seed=1336,
        device='cpu',
        catch_keyboard_interrupt=True,
    )

    dataset = ReviewDataset.load_dataset_and_make_vectorizer(data_bundle, word_vectors, unique_labels, ReviewSequenceVectorizer)
    vectorizer = dataset.get_vectorizer()
    classifier = RNN(args.input_size, args.hidden_size, args.output_size, args.num_layers)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    loss_func = nn.BCEWithLogitsLoss()

    # train
    for epoch_index in range(args.num_epoch):
        print("Epoch: ({}/{})".format(epoch_index + 1, args.num_epoch))
        dataset.set_split('train')

        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()

            y_pred = classifier(x=batch_dict['x_data'].float())

            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            loss.backward()

            optimizer.step()

            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        print('Training Accuracy', running_acc)

        # test on development set
        dataset.set_split('dev')
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0
        running_acc = 0
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = classifier(x=batch_dict['x_data'].float())

            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            loss.backward()

            optimizer.step()

            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        print('Dev Accuracy', running_acc)


    # predict on test set
    results = []
    for line in test_exs:
        padded_line = np.asarray(pad_to_length(np.array(line.indexed_words), seq_max_len), dtype=int)
        line.label = predict_rating(padded_line, classifier, vectorizer)
        results.append(line)
    return results
