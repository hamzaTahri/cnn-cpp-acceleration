import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from PIL import Image

imagefile = '../data/t10k-images-idx3-ubyte'

imagearray = idx2numpy.convert_from_file(imagefile)

print(imagearray.size)
for i in range(imagearray.size):
    im = Image.fromarray(imagearray[i])
    im.save('./images/train_image'+str(i)+'.png')
