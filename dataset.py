from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import os
import cv2


class XrayDataset(Dataset):
    def __init__(self, image_dir='data/train/JPEGImages', mask_dir='data/mask_train', size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = os.listdir(self.image_dir)

        self.transform = Compose(
            [
                ToTensor(),
                Resize(size=size),
                Normalize(mean=[0.5],
                          std=[0.5])
            ]
        )

    def __getitem__(self, item):
        image_name = self.image_names[item]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

    def __len__(self):

        return len(self.image_names)


if __name__ == '__main__':
    dataset = XrayDataset()
    pass
