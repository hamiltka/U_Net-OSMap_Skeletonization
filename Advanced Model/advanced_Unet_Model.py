import torch
import torch.nn as nn
import torchvision

def center_crop(enc_feat, target_feat):
    _, _, h, w = target_feat.shape
    enc_feat = torchvision.transforms.CenterCrop([h, w])(enc_feat)
    return enc_feat

class UNetMultiTask(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetMultiTask, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
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

        # Decoder
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

        # Multi-task output heads
        self.final_skeleton = nn.Conv2d(64, out_channels, 1)
        self.final_distance = nn.Conv2d(64, out_channels, 1)

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

        skeleton_pred = self.final_skeleton(dec1)
        distance_pred = self.final_distance(dec1)
        return skeleton_pred, distance_pred
