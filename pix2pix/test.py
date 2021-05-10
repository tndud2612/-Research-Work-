import numpy as np
import os
import scipy.misc
from train import Pix2Pix
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
     

gan = Pix2Pix()

test_model = gan.generator
test_model.load_weights("./saved_models/gen_model.h5")
path = glob("./datasets/ryan/input/*")
num = 1
for img in path:
    img_B = scipy.misc.imread(img, mode='RGB').astype(np.float)
    m,n,d = img_B.shape
    img_show = np.zeros((m,2*n,d))

    img_b = np.array([img_B])/127.5 - 1
    fake_A = 0.5* (test_model.predict(img_b))[0]+0.5
    img_show[:,:n,:] = img_B/255
    img_show[:,n:2*n,:] = fake_A
    plt.imshow(img_show)
    plt.show()
    scipy.misc.imsave("./images/output/%d.jpg" % num,img_show)
    num = num + 1
