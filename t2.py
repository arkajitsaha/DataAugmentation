import numpy as np
import torch
import torchvision
from torchvision import transforms as T
#from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import os
from PIL import Image
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# root folder is the name of the folder where data is contained
root_folder = 'imagenet-5-categories1'

train_names = sorted(os.listdir(root_folder + '/train'))
test_names = sorted(os.listdir(root_folder + '/test'))

# setting random seed to ensure the same 10% labelled data is used when training the linear classifier
random.seed(0)

names_train_10_percent = random.sample(train_names, len(train_names) // 10)
names_train = random.sample(train_names, len(train_names))
names_test = random.sample(test_names, len(test_names))

mapping = {'car': 0, 'dog': 1, 'bear': 2, 'donut': 3, 'jean': 4}
inverse_mapping = ['car', 'dog', 'bear', 'donut', 'jean']

labels_train = [mapping[x.split('_')[0]] for x in names_train]
labels_test = [mapping[x.split('_')[0]] for x in names_test]


def get_color_distortion(s=1.0):
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)

    # p is the probability of grayscale, here 0.2
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])

    return color_distort


def deprocess_and_show(img_tensor):
    return T.Compose([T.Normalize((0, 0, 0), (2, 2, 2)), T.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                      T.ToPILImage()])(img_tensor)


class MyDataset(Dataset):
    def __init__(self, root_dir, filenames, labels, mutation=False):
        self.root_dir = root_dir
        self.file_names = filenames
        self.labels = labels
        self.mutation = mutation

    def __len__(self):
        return len(self.file_names)

    def tensorify(self, img):
        res = T.ToTensor()(img)
        res = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(res)
        return res

    def mutate_image(self, img):
        res = T.RandomResizedCrop(224)(img)
        res = get_color_distortion(1)(res)
        return res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        image = T.Resize((250, 250))(image)

        if self.mutation:
            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            image2 = self.mutate_image(image)
            image2 = self.tensorify(image2)
            sample = {'image1': image1, 'image2': image2, 'label': label}
        else:
            image = T.Resize((224, 224))(image)
            image = self.tensorify(image)
            sample = {'image': image, 'label': label}

        return sample


# datasets
training_dataset_mutated = MyDataset(root_folder + '/train', names_train, labels_train, mutation=True)
testing_dataset = MyDataset(root_folder + '/test', names_test, labels_test, mutation=False)
for i in range(len(testing_dataset)):
    sample= testing_dataset[i]
c=deprocess_and_show(sample['image'])
plt.imshow(c)

for i in range(len(training_dataset_mutated)):
    sample = training_dataset_mutated[i]
    c=deprocess_and_show(sample['image1'])
    plt.imshow(c,interpolation="bicubic")
for i in range(len(training_dataset_mutated)):
    l=len(training_dataset_mutated)
sample = training_dataset_mutated[i]
c=deprocess_and_show(sample['image2'])
plt.imshow(c,interpolation="bicubic")
