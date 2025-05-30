import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, task="TB", split="train", transform=None, aug=None, class_filter=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.aug = aug
        self.class_filter = class_filter

        # Expect directory like: datasets/cleaned/TB/train or datasets/cleaned/PNEUMONIA/train
        dataset_path = os.path.join(root_dir, task.upper(), split)
        class_list = sorted(os.listdir(dataset_path))  # ['NORMAL', 'TB'] or ['NORMAL', 'PNEUMONIA']

        for label, class_name in enumerate(class_list):
            if self.class_filter is not None and label != self.class_filter:
                continue
            class_dir = os.path.join(dataset_path, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("L")

        if self.aug:
            image = np.array(image)
            image = np.expand_dims(image, axis=-1)
            image = self.aug(image=image)["image"]

        elif self.transform:
            image = self.transform(image)

        return {"image": image, "class": label}
