# Custom Dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        self.content_files = sorted(os.listdir(content_dir))
        self.style_files = sorted(os.listdir(style_dir))
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform

    def __len__(self):
        return len(self.content_files)

    def __getitem__(self, idx):
        content_path = os.path.join(self.content_dir, self.content_files[idx])
        style_path = os.path.join(self.style_dir, self.style_files[idx])

        content_image = Image.open(content_path).convert("RGB")
        style_image = Image.open(style_path).convert("RGB")

        if self.transform:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return content_image, style_image