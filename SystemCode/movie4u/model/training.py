import os
import time
from abc import ABC

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from .utils import get_attention_mask, get_token_mask


def masked_loss(y_true, y_pred, y_mask, token_mask, loss_obj):
    """
    Masked loss function.
    Only consider the mask places' predicts.
    """
    mask = tf.math.equal(y_mask, token_mask)
    loss = loss_obj(y_true, y_pred)

    mask = tf.cast(mask, loss.dtype)

    loss *= mask

    return tf.reduce_mean(loss)


def masked_accuracy(y_pred, y_true, mask):
    """
    Masked accuracy.
    Only consider the mask places' predicts.
    """
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)

    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    acc = tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))

    return acc


@tf.function
def train_step(model, ipt, target, loss_obj, mean_loss=None, mean_accuracy=None, optimizer=None, training=True):
    attn_mask = get_attention_mask(ipt)
    token_mask = get_token_mask(ipt)

    with tf.GradientTape() as tape:
        pred, _ = model(ipt, mask=attn_mask, training=training)
        loss = masked_loss(target, pred, ipt, 1, loss_obj)

    if training:
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    mean_loss(loss)
    mean_accuracy(masked_accuracy(pred, target, token_mask))


def train(model, epochs, train_dataset, val_dataset, loss_obj, optimizer):
    train_mean_loss = Mean(name='loss')
    train_mean_accuracy = Mean(name='accuracy')

    val_mean_loss = Mean(name='val_loss')
    val_mean_accuracy = Mean(name='val_accuracy')

    for epoch in range(epochs):
        start_time = time.time()

        train_mean_loss.reset_states()
        train_mean_accuracy.reset_states()

        val_mean_loss.reset_states()
        val_mean_accuracy.reset_states()

        for (batch, (ipt, target)) in enumerate(train_dataset):
            train_step(model, ipt, target, loss_obj, train_mean_loss, train_mean_accuracy, optimizer)
            if batch % 100 == 0:
                print(f'Epoch: {epoch + 1} / {epochs} Batch: {batch} ----------- '
                      f'Loss: {train_mean_loss.result():.4f} Accuracy: {train_mean_accuracy.result():.4f}')

        for (batch, (ipt, target)) in enumerate(val_dataset):
            train_step(model, ipt, target, loss_obj, val_mean_loss, val_mean_accuracy, optimizer, training=False)

        print(f'Epoch: {epoch + 1} / {epochs} Time: {time.time() - start_time:.2f}s ----------- '
              f'loss: {train_mean_loss.result():.4f} accuracy: {train_mean_accuracy.result():.4f} - '
              f'val_loss: {val_mean_loss.result():.4f} val_accuracy: {val_mean_accuracy.result():.4f}\n')
        os.makedirs(f'/content/drive/MyDrive/model/ep{epoch + 1}_loss_{train_mean_loss.result():.4f}_acc_{train_mean_accuracy.result():.4f}_val_loss_{val_mean_loss.result():.4f}_val_acc_{val_mean_accuracy.result():.4f}')

        model.save_weights(f'/content/drive/MyDrive/model/ep{epoch + 1}_loss_{train_mean_loss.result():.4f}_acc_{train_mean_accuracy.result():.4f}_val_loss_{val_mean_loss.result():.4f}_val_acc_{val_mean_accuracy.result():.4f}/model')


def evaluate(model, test_dataset, loss_obj):
    mean_loss = Mean(name='loss')
    mean_accuracy = Mean(name='accuracy')

    start_time = time.time()
    for (batch, (ipt, target)) in enumerate(test_dataset):
        train_step(model, ipt, target, loss_obj, mean_loss, mean_accuracy, training=False)

    print(f'Time: {time.time() - start_time:.2f}s ----------- '
          f'loss: {mean_loss.result():.4f} accuracy: {mean_accuracy.result():.4f}')

    return mean_loss.result(), mean_accuracy.result()


def plot_attention(attention, ipt, target, layer_name):
    fig = plt.figure(figsize=(16, 9))

    attention = tf.squeeze(attention[layer_name], 0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 2, head + 1)

        ax.matshow(attention[head][:-1, :])

        ax.set_xticks(range(len(ipt) + 2))
        ax.set_yticks(range(len(target)))

        ax.set_ylim(len(target) - 1.5, -0.5)

        ax.set_xlabel(f'Head {head + 1}')

    plt.tight_layout()
    plt.show()


class LRSchedule(LearningRateSchedule, ABC):
    """
    Implement the learning rate schedule of original paper.
    """
    def __init__(self, embedding_size, warmup_steps=4000):
        super(LRSchedule, self).__init__()

        self.embedding_size = tf.cast(embedding_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(
            tf.math.rsqrt(step), step * (self.warmup_steps ** (-1.5))
        )
