from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebATransform:
    def __init__(self, image_size: int = 128):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x)


class ImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image