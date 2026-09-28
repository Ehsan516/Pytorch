import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Loading a Dataset

training_data = datasets.FashionMNIST(
    root="data", train= True, download= True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)

test_date = datasets.FashionMNIST(
    root="data", train = False, download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)

# Iterating and Visualizing the Dataset

labels_map = { #labels to clothing items
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8)) #plot size
cols, rows = 3,3 #set the grid dimensions
for i in range(1, cols*rows +1):#matoplotlib starts at 1, repeats for the 9 grid posiitons
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #^random index from dataset and .item() to turn tensor into a regular python integer
    img, label= training_data[sample_idx]#egsts aid image and label
    figure.add_subplot(rows,cols,i)#adds axes at position in in the grid
    plt.title(labels_map[label])#displays clothing name above image, gets name from map
    plt.axis("off")#hides axes and tick marks
    plt.imshow(img.squeeze(), cmap="gray")#grayscale, .squzze turned tensor with dim of size 1 into a 28x28 array
plt.show()


# Creating a Custom Dataset for your files

#A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 

import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):

        self.img_labels = pd.read_csv(annotations_file)
        #labels csv looks like this:
        # tshirt1.jpg, 0
        # tshirt2.jpg, 0
        # ......
        # ankleboot999.jpg, 9
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    #^ len funct returns number of samples in dataset

    def __getitem__(self, idx):

        #loads and returns sample from dataset at the given index idx

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        #indentifies image's location on disk, cobverts to tensor using decode_image

        label = self.img_labels.iloc[idx, 1]
        #retries the corrsponding label from csv data

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        #returns tensor image and corrsponding label in a tuple
        return image, label


#preparing training data with DataLoaders

# pass samples in “minibatches”,
# reshuffle the data at every epoch to reduce model overfitting,
# and use Python’s multiprocessing to speed up data retrieval.
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle= True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle= True)

#Iterate through the DataLoader

# Display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")