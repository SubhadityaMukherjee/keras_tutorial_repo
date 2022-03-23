# %%
import imp
from statistics import mode
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import tensorflowjs as tfjs
from pathlib import Path
from sklearn.utils import class_weight
import numpy as np
from functools import partial

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras import layers
from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import StratifiedKFold
import utils
# %%
# Make sure everything works
print(tf.config.list_physical_devices('GPU'))
# %%
# Choices

keras.mixed_precision.set_global_policy("mixed_float16")
image_size = (64, 64)
batch_size = 32
subset = None
main_directory = Path(
    "/media/hdd/Datasets/fer2013/"
)
oversample_data = True
image_format = "png"
kfoldsplits = 5
subset = .2
exp_name = "mobilenetv2"

def label_fn(x): return x.split("/")[-2]
# %%


# load data
all_ims = [x for x in Path(main_directory).rglob(f"*{image_format}")]
print(f"Found : {len(all_ims)} files in {main_directory}")
# %%

df = utils.create_df(images=all_ims,
                     label_fn=label_fn,
                     oversample_data=True,
                     kfoldsplits=kfoldsplits,
                     force=False,  # recreate data label if you changed your data
                     )
# %%
train, val, categories = utils.split_df(
    df,
    subset=subset,
    encode_labels=True
)
# %%
print(train.head(5))
print(f"No of classes: {train.label.nunique()}")
print(train.label.value_counts())
# %%
utils.plot_some_images(train, label_fn=label_fn, n=9)
# %%
# datagen
TRAIN_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True)

TEST_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.)

train_ds = TRAIN_DATAGEN.flow_from_dataframe(dataframe=train, directory=main_directory/'train',
                                             x_col="image_id", y_col="label",
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             target_size=image_size,
                                             subset='training',
                                             )

val_ds = TEST_DATAGEN.flow_from_dataframe(dataframe=val, directory=main_directory/'test',

                                          x_col="image_id", y_col="label",
                                          batch_size=batch_size,
                                          target_size=image_size,
                                          class_mode='categorical',)
#%%
model_head = keras.applications.Xception(include_top=False, weights="imagenet")
callbacks = utils.default_callbacks(logname=exp_name)
loss_fn = keras.losses.CategoricalCrossentropy()
opt = keras.optimizers.Adam(1e-3)
metrics = ["accuracy"]

#%%
model = utils.FullModel(
    model = model_head,
    train_ds=train_ds,
    val_ds=val_ds,
    n_class=train.label.nunique(),
    image_size=image_size,
    opt=opt,
    loss_fn=loss_fn,
    metrics=metrics,
    callbacks=callbacks,
    batch_size=batch_size,
    exp_name="mobilenetv2"
)
#%%
# model.plot_model()
#%%
model.transfer_model()
print(model.model)
#%%
model.train_model(1)
#%%
#%%
# model.find_lr()
#%%