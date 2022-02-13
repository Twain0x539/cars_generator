import torch
from torch import load, nn
import cv2
import sys
from generator import GeneratorUNET

PATH_TO_GEN = '512generator'
IMAGE_SIZE = 512

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
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = torch.Tensor(image) / 255
    image = image.to(device)
    image = torch.permute(image, (2, 1, 0)).unsqueeze(0)

    output_path = 'rec_' + filename

    reconstruction = generator(image)
    reconstruction = reconstruction.squeeze().detach().cpu()
    reconstruction = torch.swapaxes(reconstruction,0,2) * 255
    reconstruction = reconstruction.numpy()
    cv2.imwrite(output_path, reconstruction)

