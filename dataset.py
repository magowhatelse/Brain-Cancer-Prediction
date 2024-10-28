import cv2 as cv
from torch.utils.data import Dataset

def read_mri(path):
    mri = cv.imread(path)
    return mri


class MRI_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = read_mri(self.dataset["Path"].iloc[idx])
        target = read_mri(self.dataset["Cancer"].iloc[idx])

        res = {
            "Name" : self.dataset["Name"].iloc[idx],
            "img" : img,
            "target" : target
        }

        return res
