import torch
from torch import load, nn
import cv2
import sys
import matplotlib.pyplot as plt

PATH_TO_GEN = 'generator_v2_20epochs_loss_13.097846984863281'
IMAGE_SIZE = 256

from torch import nn
import torch
class GeneratorUNET(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, nf=64, use_dropout=False):
        super(GeneratorUNET, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nf = nf
        self.dropout = use_dropout

        self.down_conv1 = nn.Conv2d(input_nc, nf, 4, stride=2, padding=1, bias=False)
        self.down_conv2 = nn.Conv2d(nf, nf * 2, 4, stride=2, padding=1, bias=False)
        self.down_conv3 = nn.Conv2d(nf * 2, nf * 4, 4, stride=2, padding=1, bias=False)
        self.down_conv4 = nn.Conv2d(nf * 4, nf * 8, 4, stride=2, padding=1, bias=False)
        self.down_conv5 = nn.Conv2d(nf * 8, nf * 8, 4, stride=2, padding=1, bias=False)
        self.down_conv6 = nn.Conv2d(nf * 8, nf * 8, 4, stride=2, padding=1, bias=False)
        self.down_conv7 = nn.Conv2d(nf * 8, nf * 8, 4, stride=2, padding=1, bias=False)
        self.down_bottle_conv = nn.Conv2d(nf * 8, nf * 8, 4, stride=2, padding=1, bias=False)

        self.up_bottle_conv = nn.ConvTranspose2d(nf * 8, nf * 8, 4, stride=2, padding=1, bias=False)
        self.up_conv7 = nn.ConvTranspose2d(nf * 16, nf * 8, 4, stride=2, padding=1, bias=False)
        self.up_conv6 = nn.ConvTranspose2d(nf * 16, nf * 8, 4, stride=2, padding=1, bias=False)
        self.up_conv5 = nn.ConvTranspose2d(nf * 16, nf * 8, 4, stride=2, padding=1, bias=False)
        self.up_conv4 = nn.ConvTranspose2d(nf * 16, nf * 4, 4, stride=2, padding=1, bias=False)
        self.up_conv3 = nn.ConvTranspose2d(nf * 8, nf * 2, 4, stride=2, padding=1, bias=False)
        self.up_conv2 = nn.ConvTranspose2d(nf * 4, nf, 4, stride=2, padding=1, bias=False)
        self.up_conv1 = nn.ConvTranspose2d(nf * 2, output_nc, 4, stride=2, padding=1, bias=True)

        self.up_relu = nn.ReLU(True)
        self.down_relu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()

        self.bn = nn.BatchNorm2d(nf)
        self.bn2 = nn.BatchNorm2d(nf * 2)
        self.bn4 = nn.BatchNorm2d(nf * 4)
        self.bn8 = nn.BatchNorm2d(nf * 8)

    def forward(self, x):
        x0 = self.down_relu(self.down_conv1(x))
        x1 = self.down_relu(self.bn2(self.down_conv2(x0)))
        x2 = self.down_relu(self.bn4(self.down_conv3(x1)))
        x3 = self.down_relu(self.bn8(self.down_conv4(x2)))
        x4 = self.down_relu(self.bn8(self.down_conv5(x3)))
        x5 = self.down_relu(self.bn8(self.down_conv6(x4)))
        x6 = self.down_relu(self.bn8(self.down_conv7(x5)))
        xu = self.up_relu(self.down_bottle_conv(x6))

        xu = self.bn8(self.up_bottle_conv(xu))
        xu = torch.cat([xu, x6], 1)
        xu = self.up_relu(xu)

        xu = self.bn8(self.up_conv7(xu))
        xu = torch.cat([xu, x5], 1)
        xu = self.up_relu(xu)

        xu = self.bn8(self.up_conv6(xu))
        xu = torch.cat([xu, x4], 1)
        xu = self.up_relu(xu)

        xu = self.bn8(self.up_conv5(xu))
        xu = torch.cat([xu, x3], 1)
        xu = self.up_relu(xu)

        xu = self.bn4(self.up_conv4(xu))
        xu = torch.cat([xu, x2], 1)
        xu = self.up_relu(xu)

        xu = self.bn2(self.up_conv3(xu))
        xu = torch.cat([xu, x1], 1)
        xu = self.up_relu(xu)

        xu = self.bn(self.up_conv2(xu))
        xu = torch.cat([xu, x0], 1)
        xu = self.up_relu(xu)

        xu = self.tanh(self.up_conv1(xu))

        return xu


if __name__ == "__main__":
    device='cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    try:
        generator = load(PATH_TO_GEN,map_location=torch.device(device))
    except BaseException:
        print("Generator not found")
        raise BaseException

    input_path = sys.argv[1]

    dir_list = input_path.split(sep='\\')
    filename = dir_list[-1]
    image = cv2.imread(input_path)
    image = image.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
    image = torch.Tensor(image) / 255
    image = image.to(device)
    image = torch.permute(image, (2, 1, 0)).unsqueeze(0)

    output_path = 'rec_' + filename

    reconstruction = generator(image)
    reconstruction = reconstruction.squeeze().detach().cpu()
    reconstruction = torch.swapaxes(reconstruction,0,2) * 255
    reconstruction = reconstruction.numpy()
    cv2.imwrite(output_path, reconstruction)

