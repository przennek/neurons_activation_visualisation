from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import sys
import numpy as np

model = VGG19(include_top=True, weights='imagenet')
#img = image.load_img('dragonfly.jpeg', target_size=(224, 224))
label = sys.argv[1]
img = image.load_img(label+'_label.png', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print(np.where(preds == np.max(preds)))
print('Predicted:', decode_predictions(preds, top=3)[0])

