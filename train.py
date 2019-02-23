#%matplotlib inline

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
import tkinter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from settings import *
from utils import *

# Random seed set manually for reproducibility
manualseed = 999

random.seed(manualseed)
torch.manual_seed(manualseed)

# Dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=workers)

# Choose device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2,
                                        normalize=True).cpu(), (1,2,0)))
