from keras.applications import VGG19
from keras import backend as K
import numpy as np
from PIL import Image as pimg
import cv2
import sys

def load_image(path):
    img = pimg.open("resources/doge_test_lenet.png")
    return np.array(img)

def random_np(w, h, d = 3):
    return np.random.rand(w, h, d)

def random_np_gray(w, h, d = 3):
    return np.random.rand(w, h, d) / 10 + 0.45

def numpy_to_img(numpy_array):
    return np.array([[[int(i * 255) for i in row] for row in dpt] for dpt in numpy_array])

def pimg_from_nimg(img):
    return pimg.fromarray(np.uint8(img))

def deprocess_image(x, target_std=0.1):
    # normalize tensor: center on 0., enforce std
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= target_std

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#image_resize_steps = [(32, 32), (48, 48), (64, 64), (80, 80), (128, 128), (200, 200), (256, 256), (512, 512)]
# step_iter = 0.1
image_resize_steps = [(64, 64), (128, 128), (200, 200), (256, 256), (512, 512)]
#image_resize_steps = [(256, 256)]
step_iter = 0.2

# load model and layer map
model = VGG19(include_top=False, weights='imagenet')
layers_by_name = dict([(layer.name, layer) for layer in model.layers])
input_tensor = model.inputs[0]

# create random image
img = random_np_gray(image_resize_steps[0][0], image_resize_steps[0][1])
#img = load_image("ig")
img = numpy_to_img(img)
#img = np.expand_dims(img, axis=0)

# get the filter to visualise
layer_name = 'block' + str(sys.argv[1]) + '_conv' + str(sys.argv[2])
filter_index  = int(sys.argv[3])

# calculate the loss = mean of the layer -> maximizes the activations
#output = layers_by_name[layer_name].output
#loss = K.mean(output[:, :, :, filter_index])

#gradients = K.gradients(loss, input_tensor)[0]
#gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)

# 1 step gd
#step = K.function([input_tensor], [loss, gradients])

#counter = 0
img = np.float64(img)
#for interpolation_step in image_resize_steps:
#    img = np.squeeze(img, axis=0)
#    img = cv2.resize(img, dsize=interpolation_step, interpolation=cv2.INTER_CUBIC)
#    img = np.expand_dims(img, axis=0)

#    print("Working with: " + str(image_resize_steps[counter]))
#    ran = 15 + int(image_resize_steps[counter][0] * step_iter)
#    for i in range(ran):
#        loss_val, gradients_val = step([img])
#        img += gradients_val * 1
#        print("Iteration: " + str(i) + " out of " + str(ran) )
        #if loss_val <= K.epsilon():
        #    break
#    counter += 1

#img = np.squeeze(img, axis=0)
#img = deprocess_image(img)
img = pimg_from_nimg(img)

fp = layer_name + "_" + str(filter_index) + ".png"
with open(fp, 'w+') as f:
    img.save(fp)
