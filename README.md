# Testing-out-GAN


![image](https://github.com/user-attachments/assets/f09ea436-81d6-48aa-baeb-9d7efcd01b67)


This code implements a Generative Adversarial Network (GAN) using TensorFlow/Keras to learn and generate samples from a simple 1D Gaussian distribution.  

**1. Imports and Seeds:** Imports necessary libraries (NumPy, TensorFlow, Matplotlib) and sets random seeds for reproducibility.

**2. Data Generation Functions:**

- `generate_real_samples(n)`: Generates `n` real samples from a standard normal distribution (mean=0, std=1).  Returns the samples and an array of "1" labels (indicating real data).
- `generate_latent_points(latent_dim, n)`:  Generates `n` random latent points (input noise for the generator) from a standard normal distribution with `latent_dim` dimensions.

**3. Model Definition Functions:**

- `define_generator(latent_dim)`: Defines the generator model.
    - Takes latent points as input.
    - Has a dense layer with 10 neurons and ReLU activation, followed by an output layer with a linear activation (to generate continuous values).
- `define_discriminator()`: Defines the discriminator model.
    - Takes real or generated samples as input.
    - Has a dense layer with 10 neurons and ReLU activation, followed by an output layer with a sigmoid activation (to output probabilities of real vs. fake).
    - Compiles the discriminator using binary cross-entropy loss and the Adam optimizer.
- `define_gan(generator, discriminator)`: Defines the combined GAN model.
    - Sets `discriminator.trainable = False` to freeze the discriminator's weights during generator training.
    - Connects the generator's output to the discriminator's input.
    - Compiles the GAN using binary cross-entropy loss and the Adam optimizer.

**4. Training Function (`train_gan`):**

```python
def train_gan(generator, discriminator, gan, latent_dim, n_epochs=1000, n_batch=64, n_eval=100):
    # ...
```

- `n_epochs`, `n_batch`, `n_eval`: Hyperparameters for training.
- `half_batch`: Half the batch size (used for training the discriminator on equal numbers of real and fake samples).
- The training loop iterates for `n_epochs`:
    - **Train Discriminator:**
        - Generates `half_batch` real samples and their labels.
        - Trains the discriminator on real samples (`discriminator.train_on_batch`).
        - Generates `half_batch` fake samples using the generator.
        - Creates "0" labels for the fake samples.
        - Trains the discriminator on fake samples.
    - **Train Generator:**
        - Generates `n_batch` latent points.
        - Creates "1" labels for the generated samples (tricking the discriminator into thinking they are real).
        - Trains the generator (`gan.train_on_batch`). The loss is calculated based on how well the generator can fool the discriminator.
    - **Evaluate Progress:**
        - Every `n_eval` epochs, prints the discriminator and generator losses.
        - This section has been improved to handle the possibility of the losses being lists or single values.

**5. GAN Setup and Training:**

- Sets the `latent_dim`.
- Defines the generator, discriminator, and GAN models.
- Calls `train_gan` to train the GAN.

**6. Generating and Plotting Results:**

- Generates `n` real samples.
- Generates `n` latent points and uses the trained generator to create `n` fake samples.
