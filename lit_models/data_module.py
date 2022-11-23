import glob
import random
import os

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, dst_size=(100, 400), transforms_=None, unaligned=True, mode='train'):
        if transforms_ is None:
            transforms_ = [ 
                    transforms.Resize(dst_size, Image.BICUBIC),
                    # transforms.RandomCrop(opt.size), 
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ]

        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class DataModule(pl.LightningDataModule):
    def __init__(self, root: str='data/img2real', batch_size: int=2):
        super().__init__()
        self.root = root
        self.batch_size = batch_size

    def prepare_data(self):
        import os.path
        if not os.path.exists(self.root):
            assert "No Such Files or Dirs!"

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            plates_full = ImageDataset(self.root)

            full_size = len(plates_full)
            train_size = int(full_size * 0.9)
            val_size = full_size - train_size

            self.train, self.val = random_split(plates_full, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test = ImageDataset(self.root)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
