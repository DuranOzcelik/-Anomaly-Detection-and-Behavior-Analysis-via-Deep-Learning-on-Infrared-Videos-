import torch
import torch.nn as nn

# Eğitim CONFIG'iyle aynı parametreler
_DEFAULT_CFG = {
    'encoder_filters': [32, 64, 128, 256],
    'lstm_hidden': 256,
    'lstm_layers': 2,
    'dropout': 0.3,
    'frames_per_clip': 16,
    'frame_size': 128,
}


class ConvolutionalRecurrentAutoencoder(nn.Module):
    """
    CR-AE: Convolutional Recurrent Autoencoder
    Eğitim kodundaki CRAE sınıfının birebir kopyası.

    Giriş : (batch, seq_len, 1, 128, 128)  — tek kanal (grayscale), [0, 1]
    Çıkış : (batch, seq_len, 1, 128, 128)

    Mimari: Conv2D Encoder → Linear Proj → LSTM → Linear Unproj → ConvT2D Decoder
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or _DEFAULT_CFG

        f = cfg['encoder_filters']          # [32, 64, 128, 256]
        self.spatial = f[-1] * 8 * 8        # 256 * 8 * 8 = 16384  (128 → 8 px, 4 stride-2 conv)
        lstm_h = cfg['lstm_hidden']          # 256
        drop = cfg['dropout']               # 0.3
        self.sequence_length = cfg['frames_per_clip']  # 16

        self.encoder = nn.Sequential(
            nn.Conv2d(1, f[0], 3, stride=2, padding=1), nn.BatchNorm2d(f[0]), nn.LeakyReLU(0.2),
            nn.Conv2d(f[0], f[1], 3, stride=2, padding=1), nn.BatchNorm2d(f[1]), nn.LeakyReLU(0.2),
            nn.Conv2d(f[1], f[2], 3, stride=2, padding=1), nn.BatchNorm2d(f[2]), nn.LeakyReLU(0.2),
            nn.Conv2d(f[2], f[3], 3, stride=2, padding=1), nn.BatchNorm2d(f[3]), nn.LeakyReLU(0.2),
            nn.Dropout2d(drop),
        )

        self.proj = nn.Sequential(
            nn.Linear(self.spatial, lstm_h),
            nn.LeakyReLU(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=lstm_h,
            hidden_size=lstm_h,
            num_layers=cfg['lstm_layers'],
            batch_first=True,
            dropout=drop if cfg['lstm_layers'] > 1 else 0,
        )

        self.unproj = nn.Sequential(
            nn.Linear(lstm_h, self.spatial),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(f[3], f[2], 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(f[2]), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(f[2], f[1], 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(f[1]), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(f[1], f[0], 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(f[0]), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(f[0], 1,    3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (B, T, 1, H, W)
        returns: (B, T, 1, H, W)
        """
        B, T, C, H, W = clip.shape

        x = self.encoder(clip.view(B * T, C, H, W))      # (B*T, 256, 8, 8)
        x = self.proj(x.view(B * T, -1)).view(B, T, -1)  # (B, T, 256)
        x, _ = self.lstm(x)                               # (B, T, 256)
        x = self.unproj(x.reshape(B * T, -1))             # (B*T, 16384)
        x = x.view(B * T, 256, 8, 8)
        x = self.decoder(x)                               # (B*T, 1, 128, 128)
        return x.view(B, T, 1, H, W)


# Geriye dönük uyumluluk
CRAE = ConvolutionalRecurrentAutoencoder
CR_AE = ConvolutionalRecurrentAutoencoder
