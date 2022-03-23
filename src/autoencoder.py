import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ========================
        kernel_sizes=[5,5,5,5,5]
        dropout=0.2
        batchnorm=True
        channels = [64,128,256,512,out_channels]
        for kernel_size,out_channels in zip(kernel_sizes,channels):
            padding = (kernel_size-1) // 2
            modules += [nn.Conv2d(in_channels, out_channels,
                                      kernel_size, padding=padding, stride=2)]
            modules += [nn.Dropout2d(dropout)] if dropout > 0 else []
            modules += [nn.BatchNorm2d(out_channels)] if batchnorm else []
            modules += [nn.ReLU()]
            # modules += [nn.MaxPool2d(2)] Can't use dropout due to return max indices
            in_channels = out_channels

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ========================
        kernel_sizes=reversed([5,5,5,5,5])
        dropout=0
        batchnorm=True
        channels = [512,256,128,64,out_channels]
        for i,(kernel_size,out_channels) in enumerate(zip(kernel_sizes,channels)):
            padding = (kernel_size-1) // 2
            modules += [nn.ConvTranspose2d(in_channels, out_channels,
                                      kernel_size, padding=padding, stride=2,output_padding=1)]
            
            modules += [nn.BatchNorm2d(out_channels)] if batchnorm else []
            if i<len(channels)-1:
                modules += [nn.ReLU()]
            else:
                modules += [nn.Tanh()]
            modules += [nn.Dropout2d(dropout)] if dropout > 0 else []
            # modules += [nn.MaxUnpool2d(2)] Can't use
            in_channels = out_channels


        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # Add more layers as needed for encode() and decode().
        # ========================
        self.linear_mu=nn.Linear(n_features,z_dim)
        self.linear_log_sigma2=nn.Linear(n_features,z_dim)
        self.linear_decode =nn.Linear(z_dim,n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ========================
        h = self.features_encoder(x).flatten(1)
        mu = self.linear_mu(h)
        log_sigma2 = self.linear_log_sigma2(h)
        normal_sample= torch.randn_like(mu)
        z= mu+normal_sample*torch.sqrt(torch.exp(log_sigma2)) 
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ========================
        linear_out = self.linear_decode(z).reshape(-1,*self.features_shape)
        x_rec = self.features_decoder(linear_out)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ========================
            z = torch.randn(size=(n, self.z_dim),device=device)
            samples = self.decode(z).cpu()
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ========================
    N = x.shape[0]
    x = x.flatten(1)
    xr = xr.flatten(1)
    dz = z_mu.shape[1]
    dx = x.shape[1]
    data_loss = (((x - xr) ** 2) / (x_sigma2 * dx)).sum() / N
    sigma2_x = torch.exp(z_log_sigma2)
    kldiv_loss = sigma2_x.sum(dim=1) + (z_mu ** 2).sum(dim=1) - dz - torch.log(sigma2_x).sum(dim=1)
    kldiv_loss = kldiv_loss.sum() / N
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
