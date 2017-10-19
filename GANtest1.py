import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import pylab as pyl
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2

#To stop randomnesss
seed = 128
rng = np.random.RandomState(seed)

#Set path
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'Data')

#load the data
train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))

temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)
train_x = train_x / 255.

#print an image
img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)

img = imread(filepath, flatten=True)

pyl.imshow(img, cmap='gray')
pyl.axis('off')
pyl.show()

#Define variables for later
g_input_shape = 100
d_input_shape = (28,28)
hidden_1_num_units = 500
hidden_2_num_units = 500
g_output_num_units = 784
d_output_num_units = 1
epochs = 25
batch_size = 128

#Generator network
model_1 = Sequential()
model_1.add(Dense(hidden_1_num_units, input_dim=g_input_shape, activation='relu', kernel_regularizer=L1L2(1e-5,1e-5)))
model_1.add(Dense(hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5,1e-5)))
model_1.add(Dense(g_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5,1e-5)))
model_1.add(Reshape(d_input_shape))

#Discriminator network
model_2 = Sequential()
model_2.add(InputLayer(input_shape=d_input_shape))
model_2.add(Flatten())
model_2.add(Dense(hidden_1_num_units, activation='relu', kernel_regularizer=L1L2(1e-5,1e-5)))
model_2.add(Dense(hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5,1e-5)))
model_2.add(Dense(d_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5,1e-5)))

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

gan = simple_gan(model_1, model_2, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan, player_params=[model_1.trainable_weights, model_2.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')

history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=epochs, batch_size=batch_size)

plt.plot(history.history['player_0_loss'])
plt.plot(history.history['player_1_loss'])
plt.plot(history.history['loss'])
plt.show()

zsamples = np.random.normal(size=(10,100))
pred = model_1.predict(zsamples)
for i in range(pred.shape[0]):
    plt.imshow(pred[i ,:], cmap='gray')
    plt.show()
