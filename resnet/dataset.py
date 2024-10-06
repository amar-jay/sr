from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Custom dataset to create low-resolution images from high-resolution images
class SuperResolutionVAEDataset(Dataset):
    def __init__(self, hr_images, input_px, output_px, scale_factor=2):
        self.hr_images = hr_images
        self.scale_factor = scale_factor
        self.pre_transform = transforms.Compose([
            transforms.Resize((int(output_px*1.5), int(output_px*1.5)), transforms.InterpolationMode.BICUBIC),  # Resize to a larger size if needed
            transforms.RandomCrop((output_px, output_px)),  # Crop to the desired input size
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((input_px,  input_px), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ])

        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ])


        # Define the transformation pipeline for training

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image, _ = self.hr_images[idx]
        #hr_image = Image.open(hr_image)
        pre_image = self.pre_transform(hr_image)
        hr_image = self.hr_transform(pre_image)
        lr_image = self.lr_transform(pre_image)
        return lr_image, hr_image

def get_dataloader(batch_size=64, num_workers=8, input_px=128, output_px=512):

    #TODO: write a better way to perform dataset random split
    dset = datasets.CIFAR10(root='./data', train=True, download=True)
    train_dataset = SuperResolutionVAEDataset(dset, input_px, output_px)
    val_dataset = SuperResolutionVAEDataset(datasets.CIFAR10(root='./data', train=False, download=True), input_px, output_px)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    return train_loader, val_loader

# if __name__ == '__main__':
#     train_loader, val_loader = get_dataloader(num_workers=1, batch_size=4)
#     print(len(train_loader), len(val_loader))
#     for data in train_loader:
#         inputs, labels = data
#         print(f"{inputs.shape=}")  # Check the shape to confirm the data is loaded correctly
#         print(f"{labels.shape=}")  # Check the shape to confirm the data is loaded correctly
#         break
