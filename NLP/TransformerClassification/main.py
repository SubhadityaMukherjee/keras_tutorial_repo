#%%
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

#%%
batch_size = 128
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
#%%
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
#%%
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = (
            embed_dim,
            num_heads,
            ff_dim,
            rate,
        )
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.maxlen = maxlen
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "max_len": self.maxlen,
            }
        )
        return config


#%%
def make_model(embed_dim, num_heads, ff_dim):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


# %%
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
model = make_model(embed_dim, num_heads, ff_dim)
keras.utils.plot_model(model, show_shapes=True)
# %%
epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
    keras.callbacks.ProgbarLogger(count_mode="samples", stateful_metrics=None),
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
    validation_data=(x_val, y_val),
)
#%%
# Prediction
tst = keras.preprocessing.sequence.pad_sequences([x_val[0]], maxlen=maxlen)
# %%
import numpy as np

np.argmax(model.predict(tst))
# %%
