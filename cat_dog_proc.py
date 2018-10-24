from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
from quiver_engine import server
from keras import backend as K

base_model = VGG19(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)
    print(layer)


# get_3rd_layer_output = K.function([base_model.layers[0].input],
#                                   [base_model.layers[20].output])
# img_path = 'tubingen.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# layer_output = get_3rd_layer_output([x])[0]
# print(layer_output)


def get_features(image_pre):
    base_model = VGG19(weights='imagenet', include_top=True)

    conv1_2 = K.function(base_model.layers[2])
    conv2_2 = K.function(base_model.layers[5])
    conv3_3 = K.function(base_model.layers[9])
    conv4_3 = K.function(base_model.layers[14])

    return conv1_2, conv2_2, conv3_3, conv4_3
def computer_style_loss(features):
    

