import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate real samples (a simple 1D distribution)
def generate_real_samples(n):
    return np.random.normal(0, 1, (n, 1)), np.ones((n, 1))

# Generate latent points as input for the generator
def generate_latent_points(latent_dim, n):
    return np.random.normal(0, 1, (n, latent_dim))

# Define the generator model
def define_generator(latent_dim):
    inputs = Input(shape=(latent_dim,))
    x = Dense(10, activation='relu')(inputs)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    return model

# Define the discriminator model
def define_discriminator():
    inputs = Input(shape=(1,))
    x = Dense(10, activation='relu')(inputs)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    return model

# Define the GAN model
def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Model(generator.input, discriminator(generator.output))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    return model

# Train the GAN
def train_gan(generator, discriminator, gan, latent_dim, n_epochs=1000, n_batch=64, n_eval=100):
    half_batch = n_batch // 2
    
    for i in range(n_epochs):
        # Train discriminator
        X_real, y_real = generate_real_samples(half_batch)
        d_loss_real = discriminator.train_on_batch(X_real, y_real)
        
        X_fake = generator.predict(generate_latent_points(latent_dim, half_batch))
        y_fake = np.zeros((half_batch, 1))
        d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
        
        # Train generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(X_gan, y_gan)
        
        # Evaluate progress
        if (i+1) % n_eval == 0:
            # Extract the first element if the losses are lists
            d_loss_real = d_loss_real[0] if isinstance(d_loss_real, list) else d_loss_real
            d_loss_fake = d_loss_fake[0] if isinstance(d_loss_fake, list) else d_loss_fake
            g_loss = g_loss[0] if isinstance(g_loss, list) else g_loss
            
            print(f"Epoch {i+1}, D Loss Real: {d_loss_real:.3f}, D Loss Fake: {d_loss_fake:.3f}, G Loss: {g_loss:.3f}")

# Set up the GAN
latent_dim = 2
generator = define_generator(latent_dim)
discriminator = define_discriminator()
gan = define_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, latent_dim)

# Generate and plot results
n = 1000
X_real, _ = generate_real_samples(n)
latent_points = generate_latent_points(latent_dim, n)
X_fake = generator.predict(latent_points)

plt.hist(X_real, bins=50, alpha=0.5, label='Real')
plt.hist(X_fake, bins=50, alpha=0.5, label='Generated')
plt.legend()
plt.title('Real vs Generated Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
