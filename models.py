import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


# Generator with encoder-decoder architecture and skip connections
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder layers
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # (64, H/2, W/2)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (128, H/4, W/4)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (256, H/8, W/8)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # (512, H/16, W/16)
        
        # Bottleneck layer
        self.bottleneck = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # (512, H/32, W/32)
        
        # Decoder layers
        self.dec_conv4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # (512, H/16, W/16)
        self.dec_conv3 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)  # (256, H/8, W/8)
        self.dec_conv2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)  # (128, H/4, W/4)
        self.dec_conv1 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)  # (64, H/2, W/2)
        
        # Output layer
        self.final_conv = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)  # (3, H, W)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc_conv1(x))  # (64, H/2, W/2)
        e2 = self.relu(self.enc_conv2(e1))  # (128, H/4, W/4)
        e3 = self.relu(self.enc_conv3(e2))  # (256, H/8, W/8)
        e4 = self.relu(self.enc_conv4(e3))  # (512, H/16, W/16)
        
        # Bottleneck
        b = self.relu(self.bottleneck(e4))  # (512, H/32, W/32)
        
        # Decoder with skip connections
        d4 = self.relu(self.dec_conv4(b))  # (512, H/16, W/16)
        d4 = torch.cat((d4, e4), dim=1)  # Concatenate with e4 (512 + 512 = 1024 channels)
        
        d3 = self.relu(self.dec_conv3(d4))  # (256, H/8, W/8)
        d3 = torch.cat((d3, e3), dim=1)  # (256 + 256 = 512 channels)
        
        d2 = self.relu(self.dec_conv2(d3))  # (128, H/4, W/4)
        d2 = torch.cat((d2, e2), dim=1)  # (128 + 128 = 256 channels)
        
        d1 = self.relu(self.dec_conv1(d2))  # (64, H/2, W/2)
        d1 = torch.cat((d1, e1), dim=1)  # (64 + 64 = 128 channels)
        
        # Output layer
        out = self.final_conv(d1)  # (3, H, W)
        out = torch.tanh(out)  # Output range [-1, 1]
        
        return out

# Discriminator using PatchGAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Input is (6, H, W) - concatenated content and generated images
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)  # (64, H/2, W/2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (128, H/4, W/4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (256, H/8, W/8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)  # (512, H/8-2, W/8-2)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # (1, H/8-4, W/8-4)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))  # (64, H/2, W/2)
        x = self.leaky_relu(self.conv2(x))  # (128, H/4, W/4)
        x = self.leaky_relu(self.conv3(x))  # (256, H/8, W/8)
        x = self.leaky_relu(self.conv4(x))  # (512, H/8-2, W/8-2)
        x = self.conv5(x)  # (1, H/8-4, W/8-4)
        x = torch.sigmoid(x)  # Probability map
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        
        # Extract specific layers for style and content loss
        # Exclude 'avgpool' and 'fc' layers
        self.shallow_layers = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )
        self.layer1 = resnet50.layer1  # 64
        self.layer2 = resnet50.layer2  # 128
        self.layer3 = resnet50.layer3  # 256
        self.layer4 = resnet50.layer4  # 512
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        features = []
        
        # Shallow features (after initial layers)
        x = self.shallow_layers(x)
        features.append(x)  # Features after maxpool
        
        # Deeper features
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        
        return features  # List of feature maps from different layers


class LossTracker:
    def __init__(self):
        self.content_loss_history = []
        self.style_loss_history = []
        self.adversarial_loss_history = []

    def update(self, content_loss, style_loss, adversarial_loss):
        self.content_loss_history.append(content_loss.item())
        self.style_loss_history.append(style_loss.item())
        self.adversarial_loss_history.append(adversarial_loss.item())

    def get_means(self):
        # Compute the mean of the last N losses to prevent too much fluctuation
        N = 64  # adjust N based on batch size and preferences
        content_mean = np.mean(self.content_loss_history[-N:]) if len(self.content_loss_history) >= N else np.mean(self.content_loss_history)
        style_mean = np.mean(self.style_loss_history[-N:]) if len(self.style_loss_history) >= N else np.mean(self.style_loss_history)
        adversarial_mean = np.mean(self.adversarial_loss_history[-N:]) if len(self.adversarial_loss_history) >= N else np.mean(self.adversarial_loss_history)
        return content_mean, style_mean, adversarial_mean



