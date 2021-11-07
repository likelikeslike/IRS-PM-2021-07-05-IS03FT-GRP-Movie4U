import tensorflow as tf
import numpy as np


def get_attention_mask(batch, token_pad=0):
    batch = tf.matmul(batch[..., tf.newaxis], batch[:, tf.newaxis])
    mask = tf.logical_not(tf.math.equal(batch, token_pad))
    return mask


def get_token_mask(batch, token_mask=1):
    mask = tf.math.equal(batch, token_mask)
    return mask


def predict(model, seq, tokenizer, topN=5):
    tokens = tokenizer.encode(seq)

    tokens.append(tokenizer.token_pad)

    pred, _ = model(np.array([tokens]))

    pred = pred[0][-1]

    topN_pred = np.argsort(pred)[:topN]
    topN_pred = tokenizer.decode(topN_pred)

    return topN_pred
