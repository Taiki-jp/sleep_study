import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.datasets.mnist import load_data


# PROJECTOR required log file name and address related parameters. 
LOG_DIR = './log2'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = "mnist_meta.tsv"


#  generate using the given mnist image list sprite the image. 
def create_sprite_image(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # sprite an image can be thought of as a big square matrix of all the little pictures, each of the big squares 
    #  the element is the original little picture. so the sides of this square are going to be 1, 2, 3 sqrt(n)， where n is the number of small images 。
    # np.ceil round up. np.floor round down. 
    m = int(np.ceil(np.sqrt(images.shape[0])))

    #  use all 1 to initialize the final large image 。
    sprite_image = np.ones((img_h*m, img_w*m))

    for i in range(m):
        for j in range(m):
            #  calculates the number of the current image 
            cur = i * m + j
            if cur < images.shape[0]:
                #  copy the contents of the current small image to the final sprite image 。
                sprite_image[i*img_h: (i+1)*img_h, j*img_w: (j+1)*img_w] = images[cur]

    return sprite_image


#  load mnist data 。 one is specified here _hot=False， you then get a number called labels ， represents the number represented by the current image. 
mnist = load_data()

#  generate sprite images 
to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))
sprite_image = create_sprite_image(to_visualise)

#  place the generated sprite image in the appropriate log directory 。
path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')

#  generate the corresponding label file for each image and write it to the corresponding log directory. 
path_for_mnist_metadata = os.path.join(LOG_DIR, META_FILE)
with open(path_for_mnist_metadata, 'w') as f:
    f.write('Index\tLabel\n')
    for index, label in enumerate(mnist.test.labels):
        f.write("%d\t%d\n" % (index, label))