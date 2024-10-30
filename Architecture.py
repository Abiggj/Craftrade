import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Concatenate, Reshape, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data.csv')
df = df.sort_values(by='Date')

numeric = list(df.select_dtypes(include='float64').columns)

scaler = StandardScaler()
df[numeric] = scaler.fit_transform(df[numeric])

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['news'])
vocab_size = len(tokenizer.word_index) + 1

X_text = tokenizer.texts_to_sequences(df['news'])
X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text)

tf.keras.mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for the first available GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Memory growth enabled for", gpus[0])
    except RuntimeError as e:
        print("Could not enable memory growth:", e)
else:
    print("No GPU found; running on CPU")
def build_generator(latent_dim, vocab_size, num_features):
    text_input = Input(shape=(None,))
    text_embedding = Embedding(vocab_size, latent_dim)(text_input)
    text_lstm = LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01))(text_embedding)
    text_lstm = BatchNormalization()(text_lstm)
    text_lstm = Dropout(0.2)(text_lstm)

    ohlcv_input = Input(shape=(num_features,))
    ohlcv_lstm = LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01))(tf.expand_dims(ohlcv_input, axis=-1))
    ohlcv_lstm = BatchNormalization()(ohlcv_lstm)
    ohlcv_lstm = Dropout(0.2)(ohlcv_lstm)

    merged = Concatenate()([text_lstm, ohlcv_lstm])
    output = Dense(num_features, kernel_regularizer=l2(0.01))(merged)
    output = BatchNormalization()(output)
    output = LeakyReLU(alpha=0.1)(output)
    return Model(inputs=[text_input, ohlcv_input], outputs=output)

def build_discriminator(num_features):
    model = Sequential([
        Dense(64, input_shape=(num_features,), activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = generator.input
    gan_output = discriminator(generator.output)
    model = Model(gan_input, gan_output)
    return model

def gan_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis=list(range(1, len(gradients.shape)))))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

latent_dim = 50
num_features = len(numeric)
generator = build_generator(latent_dim, vocab_size, num_features)
discriminator = build_discriminator(num_features)
gan = build_gan(generator, discriminator)

discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
generator_optimizer = Adam(learning_rate=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

discriminator.compile(optimizer=discriminator_optimizer, loss=wasserstein_loss, metrics=['accuracy'])
gan.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

epochs = 100
batch_size = 50

real_label_smoothing = 1
# discriminator_losses = []
# generator_losses = []

for epoch in range(epochs):
    num_batches = len(X_text) // batch_size
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        real_ohlcv = df[numeric].iloc[start_idx:end_idx].values
        sequential_text = X_text[start_idx:end_idx]

        dummy_numeric_input = np.zeros((batch_size, num_features), dtype=np.float32)
        generated_ohlcv = generator.predict([sequential_text, dummy_numeric_input], batch_size=batch_size)

        X_combined_batch = np.concatenate([real_ohlcv, generated_ohlcv])
        y_combined = np.concatenate([np.ones((batch_size, 1)) * real_label_smoothing, np.zeros((batch_size, 1))])

        d_loss = discriminator.train_on_batch(X_combined_batch, y_combined)
#         discriminator_losses.append(d_loss[0])

        dummy_numeric_input_full = np.zeros((batch_size, num_features), dtype=np.float32)
        fake_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch([sequential_text, dummy_numeric_input_full], fake_labels)
#         generator_losses.append(g_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")


import pickle

generator.save('./model_files/generator_model.h5')

discriminator.save('./model_files/discriminator_model.h5')

with open('./model_files/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('./model_files/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
