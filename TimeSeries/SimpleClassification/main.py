#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

#%%
batch_size = 128
root_data = "/media/hdd/Datasets/FordA/"

#%%
def readucr(filename1, filename2):
    x = np.loadtxt(filename1)
    y = np.loadtxt(filename2)
    return x, y.astype(int)


#%%
x, y = readucr(root_data + "Ford_A_train.data", root_data + "Ford_A_train.labels")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
num_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

#%%
classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()
#%%
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


# %%
model = make_model(x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)
# %%
epochs = 30

callbacks = [
    keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5", save_best_only=True),
    keras.callbacks.ProgbarLogger(count_mode="samples", stateful_metrics=None),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
loss_fn = keras.losses.SparseCategoricalCrossentropy()
opt = keras.optimizers.Adam(1e-3)

model.compile(
    optimizer=opt,
    loss=loss_fn,
    metrics=["accuracy"],
)
model.fit(
    x_train,
    y_train,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(x_test, y_test),
)
#%%
# Prediction
rand_ind = np.random.randint(len(x_test))

tst = keras.preprocessing.sequence.pad_sequences([x_test[rand_ind]])
y_test[rand_ind] == np.argmax(model.predict(tst))
# %%
