import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalRecurrentAutoencoder(nn.Module):
    """Convolutional Recurrent Autoencoder for anomaly detection via reconstruction error"""

    def __init__(self, input_channels=3, latent_dim=128, sequence_length=10):
        super(ConvolutionalRecurrentAutoencoder, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # LSTM for temporal encoding
        self.lstm_encoder = nn.LSTM(
            input_size=256 * 4 * 4,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True
        )

        # LSTM for temporal decoding
        self.lstm_decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=256 * 4 * 4,
            num_layers=2,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """
        Encode input sequence to latent representation
        x: (batch, sequence_length, channels, height, width)
        """
        batch_size, seq_len, c, h, w = x.size()

        # Encode each frame
        x_encoded = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            encoded = self.encoder(frame)
            encoded = encoded.view(batch_size, -1)
            x_encoded.append(encoded)

        x_encoded = torch.stack(x_encoded, dim=1)  # (batch, seq_len, 256*4*4)

        # LSTM encoding
        _, (h_n, c_n) = self.lstm_encoder(x_encoded)

        return h_n[-1], x_encoded

    def decode(self, latent, x_encoded):
        """
        Decode latent representation to output sequence
        latent: (batch, latent_dim)
        x_encoded: (batch, seq_len, 256*4*4)
        """
        batch_size = latent.size(0)

        # Repeat latent vector for each timestep
        latent_expanded = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Initialize hidden states for decoder LSTM
        h_0 = x_encoded[:, -1, :].unsqueeze(0).expand(2, batch_size, -1).contiguous()
        c_0 = torch.zeros(2, batch_size, 256 * 4 * 4, device=x_encoded.device)

        # LSTM decoding
        decoded, _ = self.lstm_decoder(latent_expanded, (h_0, c_0))  # (batch, seq_len, 256*4*4)

        # Decode each frame
        x_decoded = []
        for t in range(self.sequence_length):
            frame_latent = decoded[:, t, :].view(batch_size, 256, 4, 4)
            decoded_frame = self.decoder(frame_latent)
            x_decoded.append(decoded_frame)

        x_decoded = torch.stack(x_decoded, dim=1)  # (batch, seq_len, channels, height, width)

        return x_decoded

    def forward(self, x):
        """
        Forward pass through autoencoder
        x: (batch, sequence_length, channels, height, width)
        """
        latent, x_encoded = self.encode(x)
        reconstructed = self.decode(latent, x_encoded)
        return reconstructed

    def get_reconstruction_error(self, x, reduction='mean'):
        """
        Calculate MSE reconstruction error
        x: input tensor (batch, sequence_length, channels, height, width)
        returns: MSE loss per sample or mean
        """
        reconstructed = self.forward(x)

        # Calculate MSE
        mse_loss = F.mse_loss(reconstructed, x, reduction='none')

        # Reduce over all dimensions except batch
        mse_loss = mse_loss.view(mse_loss.size(0), -1).mean(dim=1)

        if reduction == 'mean':
            return mse_loss.mean()
        elif reduction == 'none':
            return mse_loss
        else:
            return mse_loss.mean()


class CR_AE(ConvolutionalRecurrentAutoencoder):
    """Alias for ConvolutionalRecurrentAutoencoder"""
    pass
