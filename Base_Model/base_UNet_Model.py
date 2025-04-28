# Ref https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/mateuszbuda_brain-segmentation-pytorch_unet.ipynb
# Ref https://www.baeldung.com/cs/neural-nets-strided-convolutions
# Ref https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html
# Ref https://www.geeksforgeeks.org/how-to-crop-an-image-at-center-in-pytorch/

import torch.nn as nn
import torch
import torchvision

def center_crop(enc_feat, target_feat):
    _, _, h, w = target_feat.shape
    enc_feat = torchvision.transforms.CenterCrop([h, w])(enc_feat)
    return enc_feat

class UNetStrided(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetStrided, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Strided conv for downsampling
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # Decoder (Expanding Path)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        bottleneck = self.bottleneck(enc3)

        dec3 = self.upconv3(bottleneck)
        enc3_crop = center_crop(enc3, dec3)
        dec3 = torch.cat((dec3, enc3_crop), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2_crop = center_crop(enc2, dec2)
        dec2 = torch.cat((dec2, enc2_crop), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1_crop = center_crop(enc1, dec1)
        dec1 = torch.cat((dec1, enc1_crop), dim=1)
        dec1 = self.dec1(dec1)

        return self.final(dec1)
