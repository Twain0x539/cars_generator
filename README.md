# edges2cars generator

![Example](https://user-images.githubusercontent.com/40434685/152781187-e99c6d0e-2f34-4b88-b222-868f6ab96bb1.png)

Image generator which uses pix2pix architecture. Converts car's edge to real car image.

# Dependencies
ipython==8.0.1

matplotlib==3.3.3

numpy==1.19.4

opencv_python==4.5.4.58

pandas==1.2.0

torch==1.10.0+cu113

torchvision==0.11.1+cu113
 

## Usage

INPUT : 512x512x3 image of edges( (255,255,255) for edge pixels, (0,0,0) otherwise)

OUTPUT : 512x512x3 reconstructed image


You can use generator by command line:

`python generate.py %edge_image.jpg%` , where %edge_image.jpg% is a path to image you want to transform 
reconstructed image will be saved to the same directory as original one with prefix "rec_"

## Related links
  1. [Pix2pix example](https://learnopencv.com/paired-image-to-image-translation-pix2pix/) 
  2. [Original Cars Dataset](https://www.kaggle.com/jessicali9530/stanford-cars-dataset/code)
  3. [Transformed Cars Dataset 512x512](https://www.kaggle.com/mihailkaraev/cars-edges-512x512?select=output512)
  4. [Kaggle notebook] (https://www.kaggle.com/mihailkaraev/cars2edges-512/notebook)

## Download
download link: https://drive.google.com/file/d/1blwwKUOR-NRgtkebG3M5pIHnbhbURmyV/view?usp=sharing

generator only(torch): https://drive.google.com/file/d/1vhWow16sAj73wEkgoRVL1FrGL36YTsdI/view 
