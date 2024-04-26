import os
from datasets.combined_dataset_base import CombinedDatasetBaseClass
from PIL import Image
from torchvision import transforms

class CombinedDatasetTest(CombinedDatasetBaseClass):
    def __init__(self, *args, **kwargs):
        super(CombinedDatasetTest, self).__init__(*args, **kwargs)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        data = dict()
        image_name_hr = self.image_names_hr[index]
        image_name_lr = self.image_names_lr[index]

        image_hr = Image.open(os.path.join(self.image_dir_hr, image_name_hr)).convert('RGB')
        image_lr = Image.open(os.path.join(self.image_dir_lr, image_name_lr)).convert('RGB')

        #TODO: check if the range is correct
        #TODO: check if to_tensor is correct for chw format and range
        image_hr_tensor = self.to_tensor(image_hr) * 2 - 1 #return a tensor in range [-1,1]
        image_lr_tensor = self.to_tensor(image_lr) * 2 - 1 #return a tensor in range [-1,1]
        
        data.update({'image_hr': image_hr_tensor, 'image_lr': image_lr_tensor})
        return data

    def __len__(self):
        return super().__len__()
