import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from anchor import normalize_annotations


class VisdroneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, 'images')
        self.anno_dir = os.path.join(root_dir, 'annotations')
        file_names = []
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                file_name = filename.split('.')[0]
                file_names.append(file_name)
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, file_name + '.jpg')
        anno_path = os.path.join(self.anno_dir, file_name + '.txt')

        image = Image.open(img_path)
        width, height = image.size

        image_rgb = Image.open(img_path).convert('RGB')
        if self.transform:
            image_rgb = self.transform(image_rgb)

        with open(anno_path, 'r') as f:
            labels = []
            for line in f.readlines():
                elems = line.strip().split(',')
                # [x, y, width, height, main_class, sub_class, ?, ?]
                label = list(map(int, elems[:]))
                labels.append(label)

        label_tensor = torch.tensor(labels)
        # transfer absolute pixel length to relative length,  x=x/width, h=h/height
        label_tensor = normalize_annotations(label_tensor, width, height)
        return image_rgb, label_tensor



def test():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = VisdroneDataset('../../task1/trainset', transform=transform)
    ddd=dataset.__getitem__(0)
    print(ddd)

# test()