import pandas as pd
import glob
from keras.callbacks import LambdaCallback
import keras.backend as K
import math

from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
import os
import tensorflowjs as tfjs
from pathlib import Path
from sklearn.utils import class_weight
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def oversample(df):
    classes = df.label.value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['label'] == key])
    classes_sample = []
    for i in range(1, len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe, classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


def create_df(images, label_fn, oversample_data=False, kfoldsplits=5, force=False):
    if not Path.exists(Path("./train_folds.csv")) or force == True:
        # get data
        df = pd.DataFrame.from_dict(
            {x: label_fn(str(x)) for x in images}, orient="index"
        ).reset_index()
        df.columns = ["image_id", "label"]
        # oversample
        if oversample_data == True:
            df = oversample(df)

        # stratified kfold
        df["kfold"] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        stratify = StratifiedKFold(n_splits=kfoldsplits)
        for i, (_, v_idx) in enumerate(
            stratify.split(X=df.image_id.values, y=df.label.values)
        ):
            df.loc[v_idx, "kfold"] = i
            df.to_csv("./train_folds.csv", index=False)
    else:
        df = pd.read_csv("./train_folds.csv")
    return df


def split_df(df, subset=None, encode_labels=True):
    df['label'] = df['label'].astype('category')
    # split into train and test
    train = df.loc[df["kfold"] != 1]
    val = df.loc[df["kfold"] == 1]

    # get a pct of the data for quick testing
    subset = int(np.floor(subset * df.shape[0]))
    # get a subset of the data
    if subset != None:
        train = train.head(subset)
        val = val.head(subset)

    print(f"Train : {len(train)}")
    print(f"Val : {len(val)}")

    if encode_labels == True:
        cats = df['label'].cat.categories
        return train, val, cats
    else:
        return train, val, None


def plot_some_images(df, label_fn, n=9):
    plt.cla()
    plt.clf()
    ims = [x for x in df.head(n).image_id.values]
    labels = [label_fn(x) for x in ims]
    f = plt.figure()
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        plt.imshow(Image.open(ims[i]))

        plt.axis('off')
    plt.show(block=True)


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
        x_train = yield from x_train
        print(x_train.shape)
        # y_train = yield from y_train
        num_batches = epochs * x_train.shape[0] / batch_size
        self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)
        print(self.lr_mult)

        # Save weights into a file
        self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(
            on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       callbacks=[callback])

        # Restore the weights to the state before model fitting
        self.model.load_weights('tmp.h5')

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end],
                 self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end],
                 derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)


class FullModel:
    def __init__(self, model, train_ds, val_ds, n_class, image_size, opt, loss_fn, metrics, callbacks, batch_size=32, exp_name="Model"):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.opt = opt
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.exp_name = exp_name
        self.n_class = n_class
        self.image_size = image_size
        self.lrfinder = LRFinder(self.model)
        self.compiled = False

    def plot_model(self):
        plt.cla()
        plt.clf()
        return keras.utils.plot_model(self.model, show_shapes=True)
    
    def find_lr(self):
        self.lrfinder.find(self.train_ds,self.val_ds, start_lr=0.0000001, end_lr=100, batch_size=self.batch_size, epochs=5)
        self.lrfinder.plot_loss(n_skip_beginning=20, n_skip_end=5)
        plt.show()

    def transfer_model(self):
        # Freeze the pretrained weights
        base_model = self.model
        self.model = None
        base_model.trainable = False

        inputs = layers.Input(shape=(None, None, 3))

        x = base_model(inputs, training = False)

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        # x = layers.BatchNormalization()(x)

        # top_dropout_rate = 0.2
        # x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(
            self.n_class, activation="softmax", name="pred")(x)

        # Compile
        base_model = tf.keras.Model(inputs, outputs, name=self.exp_name)
        self.model = base_model
        self.model.compile(
            optimizer=self.opt,
            loss=self.loss_fn,
            metrics=self.metrics
        )
        self.compiled = True

    def train_model(self, epochs, resume_training = False):
        if resume_training == True:
            try:
                last_model = os.listdir(f"./logs/{self.exp_name}/")
                last_model.sort()
                self.model.load_weights(last_model[-1])
            except:
                print("not found model")
                pass
        if self.compiled == False:
            self.model.compile(
                optimizer=self.opt,
                loss=self.loss_fn,
                metrics=self.metrics
            )
            self.compiled = True

        self.model.fit(
            self.train_ds,
            epochs=epochs,
            callbacks=self.callbacks,
            validation_data=self.val_ds,
        )


def default_callbacks(logname):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    )

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    )

    return [
        keras.callbacks.ModelCheckpoint(f"./logs/{logname}/"),
        keras.callbacks.ProgbarLogger(
            count_mode="samples", stateful_metrics=None),
        lr_scheduler,
        early_stopping
    ]
