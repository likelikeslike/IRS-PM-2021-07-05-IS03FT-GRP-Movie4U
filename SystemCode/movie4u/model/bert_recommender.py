from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization, Layer, MultiHeadAttention
from tensorflow.keras.models import Sequential, Model


def get_position_embedding(seq_len, embedding_size):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embedding_size)[np.newaxis, :]

    angles = pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_size)))

    sines = np.sin(angles[:, 0::2])
    cosines = np.cos(angles[:, 1::2])

    position_embedding = np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...]

    return tf.cast(position_embedding, tf.float32)


def feedforward_network(feedforward_size, embedding_size, activation, idx, net_type):
    return Sequential([
        Dense(feedforward_size, activation=activation, name=f'{net_type}_Block_{idx}_Feedforward_1'),
        Dense(embedding_size, name=f'{net_type}_Block_{idx}_Feedforward_2')
    ])


class BertRecommenderConfig(object):
    """
    Config class.
    :param
    item_size: the size of movie, should include pad, mask and unk token
    max_len: max len of input sequence
    embedding_size: embedding size of self-attention output
    n_encoder_block: number of encoder block
    n_head: number of head in Multi Head Attention
    feedforward_size: number of the units of the first layer in feed forward network, usually is 4 times embedding_size
    activation: activation function of feed forward network
    dropout_rate: dropout rate
    """
    def __init__(self,
                 item_size,
                 max_len=50,
                 embedding_size=128,
                 n_encoder_blocks=3,
                 n_heads=4,
                 feedforward_size=128 * 4,
                 activation='gelu',
                 dropout_rate=0.4,
                 ):
        self.item_size = item_size
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.n_encoder_blocks = n_encoder_blocks
        self.n_heads = n_heads
        self.feedforward_size = feedforward_size
        self.activation = activation
        self.dropout_rate = dropout_rate


class EncoderLayer(Layer):
    """
    Implement the Encoder Layer.
    x -> Multi Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    """
    def __init__(self, config, block_idx, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.multi_head_attention = MultiHeadAttention(
            config.n_heads, config.embedding_size, name=f'Encoder_Block_{block_idx}_MultiHeadAttention'
        )

        self.feedforward_network = feedforward_network(
            config.feedforward_size, config.embedding_size, config.activation, block_idx, 'Encoder'
        )

        self.layer_norm_1 = LayerNormalization(epsilon=1e-6, name=f'Encoder_Block_{block_idx}_LayerNormalization_1')
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6, name=f'Encoder_Block_{block_idx}_LayerNormalization_2')

        self.dropout_1 = Dropout(config.dropout_rate, name=f'Encoder_Block_{block_idx}_Dropout_1')
        self.dropout_2 = Dropout(config.dropout_rate, name=f'Encoder_Block_{block_idx}_Dropout_2')

    def call(self, x, mask=None, training=False):
        attention, attention_score = self.multi_head_attention(
            query=x, value=x, attention_mask=mask, return_attention_scores=True)
        attention = self.dropout_1(attention, training=training)
        layer_norm_output_1 = self.layer_norm_1(x + attention)

        feedforward_output = self.feedforward_network(layer_norm_output_1)
        feedforward_output = self.dropout_2(feedforward_output, training=training)
        layer_norm_output_2 = self.layer_norm_2(layer_norm_output_1 + feedforward_output)

        return layer_norm_output_2, attention_score


class EncoderModel(Layer):
    """
    Implement the Encoder Block.
    x -> Embedding + Position Embedding -> Encoder Layer
    """
    def __init__(self, config, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.config = config

        self.embedding = Embedding(
            self.config.item_size, self.config.embedding_size, mask_zero=True, name='Encoder_Embedding'
        )

        self.position_embedding = get_position_embedding(self.config.max_len, self.config.embedding_size)

        self.dropout = Dropout(self.config.dropout_rate, name='Encoder_Dropout')

        self.encoder_layers = [EncoderLayer(config, i) for i in range(1, self.config.n_encoder_blocks + 1)]

    def call(self, x, return_attention=False, mask=None, training=False):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.config.embedding_size, tf.float32))
        x += self.position_embedding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attention_scores = {}
        for i in range(self.config.n_encoder_blocks):
            x, attention_score = self.encoder_layers[i](x, mask=mask, training=training)
            attention_scores[f'Encoder_Layer_{i + 1}_Attention'] = attention_score

        if return_attention:
            return x, attention_scores
        else:
            return x


class BertRecommender(Model, ABC):
    """
    Bert Recommender
    """
    def __init__(self, config, **kwargs):
        super(BertRecommender, self).__init__(**kwargs)

        self.encoder = EncoderModel(config, name='BR_Encoder')

        self.final_dense = Dense(config.item_size, name='BR_Dense')

    def call(self,
             inputs,
             mask=None,
             training=False
             ):

        encoder_output, attention = self.encoder(inputs, return_attention=True, mask=mask, training=training)

        output = self.final_dense(encoder_output)

        return output, attention
