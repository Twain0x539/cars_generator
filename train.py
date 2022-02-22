import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import GeneratorUNET, PatchDicriminator

from os import listdir
from os.path import isfile, join

from IPython.display import clear_output

import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision import transforms

import cv2


RESIZE = 512
RANDOM_SEED = 777
BATCH_SIZE = 10

BASE_DIR = '../input/cars-edges-512x512/'
INPUT_DIR = BASE_DIR + f'input{RESIZE}/'
OUTPUT_DIR = BASE_DIR + f'output{RESIZE}/'


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class CarsDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        filenames = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
        self.filenames = pd.Series(filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = cv2.imread(self.root_dir + self.filenames[idx])
        image = torch.Tensor(image) / 255
        image = torch.permute(image, (2, 1, 0))
        return image


def gen_loss(gen_img, real_img, y_pred, y_real):
    bce = nn.BCELoss()
    l1loss = nn.L1Loss()

    return bce(y_pred, y_real) + 100 * l1loss(gen_img, real_img)

input_data = CarsDataset(INPUT_DIR)
output_data = CarsDataset(OUTPUT_DIR)

input_dl = DataLoader(input_data,batch_size=BATCH_SIZE)
output_dl = DataLoader(output_data, batch_size=BATCH_SIZE)


disc_loss = nn.BCELoss()
generator = GeneratorUNET().to(device)
discriminator = PatchDicriminator().to(device)
gen_optim = torch.optim.Adam(generator.parameters(),lr=0.0002, betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(),lr=0.0002)



##################### Training ####################
epochs = 11

D_losses = []
G_losses = []

for epoch in np.arange(epochs):
    D_epoch_losses = []
    G_epoch_losses = []
    print(f'epoch {epoch} / {epochs}')
    if (epoch > 0):
        print(f"Discriminator loss : {D_losses[-1]}, Generator loss : {G_losses[-1]}")
        if not (epoch % 5):
            torch.save(generator, f'./NEW512generator_v2_{epoch + 30} epochs loss {G_losses[-1]}')

    for contour, real_img in zip(input_dl, output_dl):
        contour = contour.to(device)
        real_img = real_img.to(device)
        disc_optim.zero_grad()

        generated_images = generator(contour).to(device)

        real_labels = Variable(torch.ones(contour.shape[0], 1, 30, 30)).to(device)
        fake_labels = Variable(torch.zeros(contour.shape[0], 1, 30, 30)).to(device)

        fake_cat = torch.cat([contour, generated_images], 1)
        # nije fake_cat.detach()!!!!
        fake_images_preds = discriminator(fake_cat.detach())
        fake_images_loss = disc_loss(fake_images_preds, fake_labels)

        real_cat = torch.cat([contour, real_img], 1)
        real_images_preds = discriminator(real_cat)
        real_images_loss = disc_loss(real_images_preds, real_labels)

        total_loss = (real_images_loss + fake_images_loss) / 2
        D_epoch_losses.append(total_loss)
        total_loss.backward(retain_graph=False)
        disc_optim.step()

        gen_optim.zero_grad()

        gen_cat = torch.cat([contour, generated_images], 1)
        pred_labels = discriminator(gen_cat)
        g_loss = gen_loss(generated_images, real_img, pred_labels, real_labels)
        G_epoch_losses.append(g_loss)
        g_loss.backward()
        gen_optim.step()

    D_losses.append(sum(D_epoch_losses) / len(D_epoch_losses))
    G_losses.append(sum(G_epoch_losses) / len(G_epoch_losses))