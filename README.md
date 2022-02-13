# edges2cars generator

![Example](https://user-images.githubusercontent.com/40434685/152781187-e99c6d0e-2f34-4b88-b222-868f6ab96bb1.png)

Image generator which uses pix2pix architecture. Converts car's edge to real car image.

# Dependencies
torch,
cv2

## Usage

INPUT : 512x512x3 image of edges( (255,255,255) for edge pixels, (0,0,0) otherwise)

OUTPUT : 512x512x3 reconstructed image



You can use generator by command line:

`python main.py %edge_image.jpg%` , where %edge_image.jpg% is a path to image you want to transform 
reconstructed image will be saved to the same directory as original one with prefix "rec_"

## Related links
  1. [Pix2pix example](https://learnopencv.com/paired-image-to-image-translation-pix2pix/) 
  2. [Original Cars Dataset](https://www.kaggle.com/jessicali9530/stanford-cars-dataset/code)
  3. [Transformed Cars Dataset 512x512]()


## Download
download link: https://drive.google.com/file/d/1blwwKUOR-NRgtkebG3M5pIHnbhbURmyV/view?usp=sharing
