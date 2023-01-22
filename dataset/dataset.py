import csv 
import numpy as np
from PIL import Image 
from torch.utils.data import Dataset
from config.config import *

class AttributesDataset():
    def __init__(self, annot_path) -> None:
        color_labels = []
        gender_labels = []
        article_labels = []
        
        with open(annot_path) as file:
            fashion = csv.DictReader(file)
            for row in fashion:
                color_labels.append(row['baseColour'])
                gender_labels.append(row['gender'])
                article_labels.append(row['articleType'])
        self.color_labels = np.unique(color_labels)
        self.gender_labels = np.unique(gender_labels)
        self.article_labels = np.unique(article_labels)
        
        self.num_colors = len(self.color_labels)
        self.num_genders = len(self.gender_labels)
        self.num_articles = len(self.article_labels)
        
        self.color_id_to_name = dict(
            zip(range(len(self.color_labels)), self.color_labels))
        self.color_name_to_id = dict(
            zip(self.color_labels, range(len(self.color_labels))))

        self.gender_id_to_name = dict(
            zip(range(len(self.gender_labels)), self.gender_labels))
        self.gender_name_to_id = dict(
            zip(self.gender_labels, range(len(self.gender_labels))))

        self.article_id_to_name = dict(
            zip(range(len(self.article_labels)), self.article_labels))
        self.article_name_to_id = dict(
            zip(self.article_labels, range(len(self.article_labels))))

class FashionDataset(Dataset):
    def __init__(self, annot_path, attributes, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.attr = attributes
        
        self.data = []
        self.color_labels = []
        self.gender_labels = []
        self.article_labels = []
        
        with open(annot_path) as file:
            fashion = csv.DictReader(file)
            for row in fashion:
                self.data.append(row['image_path'])
                self.color_labels.append(
                    self.attr.color_name_to_id[row['baseColour']]
                )
                self.gender_labels.append(
                    self.attr.gender_name_to_id[row['gender']]
                )
                self.article_labels.append(
                    self.attr.color_articleType_to_id[row['articleType']]
                )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data[index]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        fashion_attr = dict(image=image,
            labels=dict(
                color_labels=self.color_labels[index],
                gender_labels=self.gender_labels[index],
                article_labels=self.article_labels[index]
            )
        )
        return fashion_attr