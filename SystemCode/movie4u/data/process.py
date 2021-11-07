from ast import literal_eval

import numpy as np
import pandas as pd
import tensorflow as tf

PAD = 0
MASK = 1
UNK = 2


def create_masked_seq(tokens, n_items, masked_prob=0.15, max_mask_count=5):
    token_indexes = []

    for i, token in enumerate(tokens):
        if token == 2:
            continue
        token_indexes.append(i)

    np.random.shuffle(token_indexes)
    output_tokens = list(tokens)
    num_to_mask = min(max_mask_count, max(1, int(round(len(tokens) * masked_prob))))

    masked_items = []
    covered_indexes = set()

    for index in token_indexes:
        if len(masked_items) >= num_to_mask:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        if np.random.random() < 0.8:
            masked_token = MASK
        else:
            if np.random.random() < 0.5:
                masked_token = tokens[index]
            else:
                masked_token = np.random.randint(2, n_items - 1)

        output_tokens[index] = masked_token
        masked_items.append((index, tokens[index]))

    return output_tokens


def split_seq(seq, max_len=50, max_offset=10, max_seq_count=5, min_len=10):
    seq_len = len(seq)

    if seq_len <= max_len:
        return [seq]

    result = []
    split_seq_len = np.random.randint(max_len - max_offset, max_len, max_seq_count - 1)
    split_indexes = [sum(split_seq_len[:i]) for i in range(1, max_seq_count)]

    curr_index = 0
    for index in split_indexes:
        if index > seq_len + 1:
            return result
        result.append(seq[curr_index: index])
        curr_index = index

    if seq_len - curr_index >= min_len:
        result.append(seq[curr_index:])

    return result


def get_dataset(data, item_size, max_len=50, max_offset=10, max_seq_count=5, min_len=10, shuffle=False):
    X, Y = [], []
    i = 1
    for d in data:
        print(f'{i} / {data.shape[0]}', end='\r')
        i += 1
        seq = literal_eval(d)
        if shuffle:
            np.random.shuffle(seq)

        splitted_seq = split_seq(seq, max_len, max_offset, max_seq_count, min_len)

        for s in splitted_seq:
            x = create_masked_seq(s, item_size)

            X.append(str(x))
            Y.append(str(s))

    dataset = tf.data.TextLineDataset.from_tensor_slices((X, Y))
    return dataset


def split_dataset(train_filepath,
                  test_filepath,
                  item_size,
                  train_val_split=0.8,
                  batch_size=32,
                  max_len=50,
                  max_offset=10,
                  max_seq_count=5,
                  min_len=10,
                  shuffle=False):
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    train_data = train_df['movieIndexes'].values
    np.random.shuffle(train_data)

    test_data = test_df['movieIndexes'].values
    np.random.shuffle(test_data)

    train_data = train_data[:int(train_data.shape[0] * train_val_split)]
    val_data = train_data[int(train_data.shape[0] * train_val_split):]

    train_dataset = get_dataset(train_data, item_size, max_len, max_offset, max_seq_count, min_len, shuffle)
    val_dataset = get_dataset(val_data, item_size, max_len, max_offset, max_seq_count, min_len, shuffle)
    test_dataset = get_dataset(test_data, item_size, max_len, max_offset, max_seq_count, min_len, shuffle)

    train_dataset = train_dataset.map(tf_mask_dataset)
    train_dataset = train_dataset.shuffle(train_data.shape[0] // 2).padded_batch(batch_size, padded_shapes=([-1], [-1]))

    val_dataset = val_dataset.map(tf_mask_dataset)
    val_dataset = val_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]))

    test_dataset = test_dataset.map(tf_mask_dataset)
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]))

    return train_dataset, val_dataset, test_dataset


def mask_dataset(x, y):
    x = literal_eval(x.numpy().decode('utf-8'))
    y = literal_eval(y.numpy().decode('utf-8'))

    return x, y


def tf_mask_dataset(x, y):
    return tf.py_function(mask_dataset, [x, y], [tf.int64, tf.int64])


class Tokenizer(object):

    def __init__(self, token_dict_path, token_pad=0, token_mask=1, token_unk=2):
        self.__encode_dict = {}
        self.__decode_dict = {}

        self.token_pad = token_pad
        self.token_mask = token_mask
        self.token_unk = token_unk

        self.build_encode_decode_dict(token_dict_path)

        self.item_size = len(self.__encode_dict)

    def build_encode_decode_dict(self, path):
        with open(path, 'r') as f:
            for line in f:
                idx, token = line.strip().split(' ')
                self.__encode_dict[token] = int(idx)
                self.__decode_dict[int(idx)] = token

    def encode(self, seq):
        ids = []
        for token in seq:
            try:
                ids.append(self.__encode_dict[str(token)])
            except KeyError:
                ids.append(self.token_unk)

        return ids

    def decode(self, seq):
        return [self.__decode_dict[int(idx)] for idx in seq]
