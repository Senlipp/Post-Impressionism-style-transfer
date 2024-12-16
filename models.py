import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder layers (unchanged)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)    # (64, H/2, W/2)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (128, H/4, W/4)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # (256, H/8, W/8)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # (512, H/16, W/16)
        
        # Bottleneck layer
        self.bottleneck = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1) # (512, H/32, W/32)
        
        # Decoder
        # Use reflection padding before each conv to avoid hard edges
        # Remove padding=1 from convs since reflection pad handles that.
        
        # Decoder stage corresponding to enc_conv4
        self.dec_up4 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.dec_pad4 = nn.ReflectionPad2d(1)
        self.dec_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.dec_norm4 = nn.InstanceNorm2d(512, affine=True)
        
        # Decoder stage corresponding to enc_conv3
        self.dec_up3 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.dec_pad3 = nn.ReflectionPad2d(1)
        self.dec_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=0)
        self.dec_norm3 = nn.InstanceNorm2d(256, affine=True)
        
        # Decoder stage corresponding to enc_conv2
        self.dec_up2 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.dec_pad2 = nn.ReflectionPad2d(1)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=0)
        self.dec_norm2 = nn.InstanceNorm2d(128, affine=True)
        
        # Decoder stage corresponding to enc_conv1
        self.dec_up1 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.dec_pad1 = nn.ReflectionPad2d(1)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
        self.dec_norm1 = nn.InstanceNorm2d(64, affine=True)
        
        # Final output layer
        self.final_up = nn.Upsample(scale_factor=2, mode='bicubic')
        self.final_pad = nn.ReflectionPad2d(1)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=0)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        
        # Learnable weights for skip connections
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.alpha4 = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # -----------------
        #      Encoder
        # -----------------
        e1 = self.relu(self.enc_conv1(x))   # (64, H/2, W/2)
        e2 = self.relu(self.enc_conv2(e1))  # (128, H/4, W/4)
        e3 = self.relu(self.enc_conv3(e2))  # (256, H/8, W/8)
        e4 = self.relu(self.enc_conv4(e3))  # (512, H/16, W/16)
        
        # Bottleneck
        b = self.relu(self.bottleneck(e4))  # (512, H/32, W/32)
        
        # -----------------
        #      Decoder
        # -----------------
        # Decode from bottleneck to match e4 size
        d4 = self.dec_up4(b)
        d4 = self.dec_pad4(d4)
        d4 = self.relu(self.dec_norm4(self.dec_conv4(d4)))
        d4 = self.alpha4 * d4 + (1 - self.alpha4) * e4
        
        # Decode and skip connect with e3
        d3 = self.dec_up3(d4)
        d3 = self.dec_pad3(d3)
        d3 = self.relu(self.dec_norm3(self.dec_conv3(d3)))
        d3 = self.alpha3 * d3 + (1 - self.alpha3) * e3
        
        # Decode and skip connect with e2
        d2 = self.dec_up2(d3)
        d2 = self.dec_pad2(d2)
        d2 = self.relu(self.dec_norm2(self.dec_conv2(d2)))
        d2 = self.alpha2 * d2 + (1 - self.alpha2) * e2
        
        # Decode and skip connect with e1
        d1 = self.dec_up1(d2)
        d1 = self.dec_pad1(d1)
        d1 = self.relu(self.dec_norm1(self.dec_conv1(d1)))
        d1 = self.alpha1 * d1 + (1 - self.alpha1) * e1
        
        # Final upsample and output
        out = self.final_up(d1)
        out = self.final_pad(out)
        out = self.final_conv(out)
        out = torch.tanh(out)  # Range [-1, 1]
        
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
        vgg19 = models.vgg19(pretrained=True).features

        # We select the layers of interest for style and content
        # Layers (with indices):
        # relu1_1: index 1
        # relu2_1: index 6
        # relu3_1: index 11
        # relu4_1: index 20
        # relu4_2: index 21 (content)
        self.style_layers = [1, 6, 11, 20]
        self.content_layer = 21

        # Keep only layers up to relu4_2 for efficiency if desired:
        # or just keep all and stop after reaching content_layer
        self.vgg_layers = vgg19[:self.content_layer+1]

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        style_features = []
        content_feature = None

        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.style_layers:
                style_features.append(x)
            if i == self.content_layer:
                content_feature = x
                break

        # Return style and content features
        return style_features, content_feature






