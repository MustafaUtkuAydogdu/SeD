import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
import math

class CombinedDataset(Dataset):
    def __init__(self, mirror_augment_prob=0.5, image_size = 256):
        self.num_samples = len(self.image_names)
        self.image_size = image_size
        self.mirror_augment_prob = mirror_augment_prob

        self.image_dir_hr = "dataset_cropped_hr"
        self.image_dir_lr = "dataset_cropped_lr"
        self.image_names_hr = sorted(os.listdir(self.image_dir_hr))
        self.image_names_lr = sorted(os.listdir(self.image_dir_lr))
        self.down_sample_factor = 4
        self.down_sampled_image_size = self.image_size // self.down_sample_factor



    def preprocess_image(self, image):
        if np.random.uniform() < self.mirror_augment_prob:
            image = np.fliplr(image)

        #TODO: Check if np.asarray is necessary
        image = (image).transpose(2, 0, 1).astype(np.float32) # H,W,C -> C,H,W
        image = (image - 127.5) / 127.5 # Normalize (Range: [-1.0, 1.0])
        image = torch.tensor(image, dtype=torch.float32)
        return image
    
    def __len__(self):
        return self.num_samples
    
    def crop_image(self, image, crop_height, crop_width, hr_image = True):

        #TODO: Check if dimensions are correct
        if hr_image:
            return image[crop_height:crop_height + self.image_size, crop_width:crop_width + self.image_size, ...]
        else:
            return image[crop_height:crop_height + self.down_sampled_image_size, crop_width:crop_width + self.down_sampled_image_size, ...]
    
    def __getitem__(self, index):
        data = dict()
        image_name_hr = self.image_names_hr[index]
        image_name_lr = self.image_names_lr[index]

        image_hr = Image.open(os.path.join(self.image_dir_hr, image_name_hr)).convert('RGB')
        image_lr = Image.open(os.path.join(self.image_dir_lr, image_name_lr)).convert('RGB')

        #get height and width of the high resolution image
        hr_width, hr_height = image_hr.size
        #get height and width of the low resolution image
        lr_width, lr_height = image_lr.size

        #get the crop size for the high resolution image
        cropped_height = np.random.randint(0, hr_height - self.image_size + 1)
        cropped_width = np.random.randint(0, hr_width - self.image_size + 1)

        #get the crop size for the low resolution image
        image_hr_cropped = self.crop_image(np.array(image_hr), cropped_height, cropped_width)
        image_lr_cropped = self.crop_image(np.array(image_lr), cropped_height, cropped_width,hr_image = False)      

        image_hr_cropped = self.preprocess_image(image_hr_cropped)
        image_lr_cropped = self.preprocess_image(image_lr_cropped)

        data.update({'image_hr': image_hr_cropped, 'image_lr': image_lr_cropped})
        return data
    

    