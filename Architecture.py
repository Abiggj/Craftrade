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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime

# Read and preprocess data
df = pd.read_csv('model_files/data.csv')
df = df.sort_values(by='Date')

numeric = list(df.select_dtypes(include='float64').columns)

scaler = StandardScaler()
df[numeric] = scaler.fit_transform(df[numeric])

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['news'])
vocab_size = len(tokenizer.word_index) + 1

X_text = tokenizer.texts_to_sequences(df['news'])
X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text)

# Configure GPU settings
tf.keras.mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Memory growth enabled for", gpus[0])
    except RuntimeError as e:
        print("Could not enable memory growth:", e)
else:
    print("No GPU found; running on CPU")

# Create directories for saving plots
plots_dir = Path('./gan_plots')
plots_dir.mkdir(exist_ok=True)


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


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def plot_training_history(d_losses, g_losses, d_accuracies, save_dir):
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.title('GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot discriminator accuracy
    plt.subplot(1, 2, 2)
    plt.plot(d_accuracies, label='Discriminator Accuracy', color='green', alpha=0.7)
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(save_dir / f'training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_distribution_comparison(real_data, generated_data, feature_names, save_dir):
    n_features = real_data.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.kdeplot(data=real_data[:, i], label='Real', alpha=0.6)
        sns.kdeplot(data=generated_data[:, i], label='Generated', alpha=0.6)
        plt.title(f'{feature_names[i]} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(save_dir / f'distribution_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Initialize models
latent_dim = 50
num_features = len(numeric)
generator = build_generator(latent_dim, vocab_size, num_features)
discriminator = build_discriminator(num_features)
gan = build_gan(generator, discriminator)

# Compile models
discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
generator_optimizer = Adam(learning_rate=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

discriminator.compile(optimizer=discriminator_optimizer, loss=wasserstein_loss, metrics=['accuracy'])
gan.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

# Training parameters
epochs = 200
batch_size = 50
plot_interval = 40 # Plot every 10 epochs
real_label_smoothing = 1

# Initialize lists to store metrics
discriminator_losses = []
generator_losses = []
discriminator_accuracies = []

# Training loop
for epoch in range(epochs):
    epoch_d_losses = []
    epoch_g_losses = []
    epoch_d_accuracies = []

    num_batches = len(X_text) // batch_size
    for batch in range(num_batches):
        # Get batch data
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        real_ohlcv = df[numeric].iloc[start_idx:end_idx].values
        sequential_text = X_text[start_idx:end_idx]

        # Generate fake data
        dummy_numeric_input = np.zeros((batch_size, num_features), dtype=np.float32)
        generated_ohlcv = generator.predict([sequential_text, dummy_numeric_input], batch_size=batch_size)

        # Train discriminator
        X_combined_batch = np.concatenate([real_ohlcv, generated_ohlcv])
        y_combined = np.concatenate([np.ones((batch_size, 1)) * real_label_smoothing, np.zeros((batch_size, 1))])
        d_loss = discriminator.train_on_batch(X_combined_batch, y_combined)

        # Train generator
        dummy_numeric_input_full = np.zeros((batch_size, num_features), dtype=np.float32)
        fake_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch([sequential_text, dummy_numeric_input_full], fake_labels)

        epoch_d_losses.append(d_loss[0])
        epoch_d_accuracies.append(d_loss[1])
        epoch_g_losses.append(g_loss)

    # Calculate epoch averages
    avg_d_loss = np.mean(epoch_d_losses)
    avg_g_loss = np.mean(epoch_g_losses)
    avg_d_acc = np.mean(epoch_d_accuracies)

    discriminator_losses.append(avg_d_loss)
    generator_losses.append(avg_g_loss)
    discriminator_accuracies.append(avg_d_acc)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Discriminator Loss: {avg_d_loss:.4f}")
    print(f"Generator Loss: {avg_g_loss:.4f}")
    print(f"Discriminator Accuracy: {avg_d_acc:.4f}")

    # Generate plots at intervals
    if (epoch + 1) % plot_interval == 0:
        # Plot training history
        plot_training_history(discriminator_losses, generator_losses, discriminator_accuracies, plots_dir)

        # Generate samples for distribution comparison
        sample_size = min(1000, len(X_text))
        sample_indices = np.random.choice(len(X_text), sample_size, replace=False)
        real_samples = df[numeric].iloc[sample_indices].values
        sequential_text_samples = X_text[sample_indices]
        dummy_numeric_input = np.zeros((sample_size, num_features), dtype=np.float32)
        generated_samples = generator.predict([sequential_text_samples, dummy_numeric_input], batch_size=batch_size)

        # Plot distribution comparison
        plot_distribution_comparison(real_samples, generated_samples, numeric, plots_dir)

# Generate final plots
plot_training_history(discriminator_losses, generator_losses, discriminator_accuracies, plots_dir)

sample_size = min(1000, len(X_text))
sample_indices = np.random.choice(len(X_text), sample_size, replace=False)
real_samples = df[numeric].iloc[sample_indices].values
sequential_text_samples = X_text[sample_indices]
dummy_numeric_input = np.zeros((sample_size, num_features), dtype=np.float32)
generated_samples = generator.predict([sequential_text_samples, dummy_numeric_input], batch_size=batch_size)
plot_distribution_comparison(real_samples, generated_samples, numeric, plots_dir)

print("Training completed. Plots have been saved in the 'gan_plots' directory.")
#
#
# import pickle
#
# generator.save('./model_files/generator_model.h5')
#
# discriminator.save('./model_files/discriminator_model.h5')
#
# with open('./model_files/tokenizer.pkl', 'wb') as f:
#     pickle.dump(tokenizer, f)
#
# with open('./model_files/scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
