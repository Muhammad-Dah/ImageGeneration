import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ========================
        out_channels = 256
        self.feature_extractor = nn.Sequential(nn.Conv2d(in_size[0], 64, 5, padding=2, stride=2), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                                               nn.Dropout2d(0.25),
                                               nn.Conv2d(64, 128, 5, padding=2, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.Dropout2d(0.25),
                                               nn.Conv2d(128, 256, 5, padding=2, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
                                               nn.Dropout2d(0.25),
                                               nn.Conv2d(256, 512, 5, padding=2, stride=2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
                                               nn.Dropout2d(0.25))
        self.discriminator = nn.Linear(512 * in_size[1] * in_size[2] // (16 * 16), 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        #  Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ========================
        features = self.feature_extractor(x).flatten(1)
        y = self.discriminator(features)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :feature_map_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ========================
        self.feature_map_size = featuremap_size
        self.W = nn.Linear(z_dim, 1024 * featuremap_size ** 2)
        self.model = nn.Sequential(nn.BatchNorm2d(1024), nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
                                   nn.ConvTranspose2d(512, 256, 5, 2, 2, 1), nn.BatchNorm2d(
                256), nn.ReLU(),
                                   nn.ConvTranspose2d(256, 128, 5, 2, 2, 1), nn.BatchNorm2d(
                128), nn.ReLU(),
                                   nn.ConvTranspose2d(128, 3, 5, 2, 2, 1), nn.BatchNorm2d(3), nn.Tanh())
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        #  Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ========================
        if with_grad:
            z = torch.randn((n, self.z_dim), device=device)
            samples = self.forward(z)
        else:
            with torch.no_grad():
                z = torch.randn((n, self.z_dim), device=device)
                samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        #  Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ========================
        w = self.W
        h = w(z).reshape(-1, 1024, self.feature_map_size, self.feature_map_size)
        x = self.model(h)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ========================
    noise = torch.rand_like(y_data) * label_noise - label_noise / 2
    data = data_label + noise
    loss_data = torch.nn.functional.binary_cross_entropy_with_logits(
        y_data, data)
    generated_label = (1 - data_label) + torch.rand_like(
        y_generated) * label_noise - label_noise / 2
    loss_generated = torch.nn.functional.binary_cross_entropy_with_logits(
        y_generated, generated_label)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ========================
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_generated, torch.full_like(y_generated, data_label))
    # ========================
    return loss


def train_batch(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        gen_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        gen_optimizer: Optimizer,
        x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ========================
    dsc_optimizer.zero_grad()
    data = x_data
    N = data.shape[0]
    generated_data = gen_model.sample(N, with_grad=True)
    y_data = dsc_model(data)
    y_generated = dsc_model(generated_data.detach())
    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    #  Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ========================
    gen_optimizer.zero_grad()
    y_generated = dsc_model(generated_data)
    gen_loss = gen_loss_fn(y_generated)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ========================
    losses = [dsc_loss + gen_loss for dsc_loss, gen_loss in zip(dsc_losses, gen_losses)]
    if dsc_losses[-1] + gen_losses[-1] <= min(losses):
        saved = True

    if saved and checkpoint_file is not None:
        torch.save(gen_model, checkpoint_file)
        print(f'*** Saved checkpoint {checkpoint_file} ')
    # ========================

    return saved
