import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

class AnimeCharacterDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file with annotations
            img_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Convert string index to actual filename
        self.data['filename'] = self.data['index'].apply(lambda x: x.split('.txt')[0] + '.jpg')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        # Get all labels
        labels = torch.tensor([
            self.data.iloc[idx]['Age'],
            self.data.iloc[idx]['Gender'],
            self.data.iloc[idx]['Ethnicity'],
            self.data.iloc[idx]['Hair Style'],
            self.data.iloc[idx]['Hair Color'],
            self.data.iloc[idx]['Hair Length'],
            self.data.iloc[idx]['Eye Color'],
            self.data.iloc[idx]['Body Type'],
            self.data.iloc[idx]['Dress']
        ], dtype=torch.float32)
        
        return image, labels


if __name__ == "__main__":
    dataset = AnimeCharacterDataset(
        csv_path='train_data.csv',
        img_dir='single_characters'
    )
    print(dataset.__getitem__(0))
