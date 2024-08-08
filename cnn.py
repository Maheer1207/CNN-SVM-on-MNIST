import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import fashion_mnist, mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten
from keras import backend as k

def cnn_mnist(res):
  if (res == "original"):
    x_train, y_train, x_test, y_test = load("fashion")
  elif (res == "low"):
    x_train, y_train, x_test, y_test = load("original")
  
  x_train, y_train, x_test, y_test, input_x = process(x_train, y_train, x_test, y_test)

  if (res == "original"):
    model = build_org_res(input_x)
  elif (res == "low"):
    x_train, x_test, input_x = down_sample(x_train, x_test, input_x)
    model = build_low_res(input_x)

  train_fit(model, x_train, y_train, x_test, y_test)
  return model, x_train, y_train, x_test, y_test

def load(type):
  if (type == "fashion"):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  elif (type == "original"):
     (x_train, y_train), (x_test, y_test) = mnist.load_data()

  return x_train, y_train, x_test, y_test

def process(x_train, y_train, x_test, y_test):
  # Reshape images to fit the CNN input requirement
  img_rows, img_cols = 28, 28

  if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_x = (1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_x = (img_rows, img_cols, 1)

  # Normalize pixel values to be between 0 and 1
  x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
  y_train, y_test = keras.utils.to_categorical(y_train), keras.utils.to_categorical(y_test)
  return x_train, y_train, x_test, y_test, input_x

def build_org_res(input_x):
  input_x_org = Input(shape=input_x)
  layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_x_org)
  layer2 = MaxPooling2D((2, 2))(layer1)
  layer3 = Conv2D(64, (3, 3), activation='relu')(layer2)
  layer4 = MaxPooling2D((3, 3))(layer3)  # Changed to match your new structure
  layer5 = Dropout(0.5)(layer4)  # Dropout layer
  layer6 = Flatten()(layer5)
  layer7 = Dense(250, activation='sigmoid')(layer6)  # Changed Dense layer size and activation
  output_original = Dense(10, activation='softmax')(layer7)

  model_org = Model([input_x_org], output_original)
  model_org.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.categorical_crossentropy,
                metrics=['accuracy'])
  return model_org

def down_sample(x_train, x_test, input_x):
  # Downsample images using average pooling
  input_layer_org = Input(shape=input_x)
  output_layer_low = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer_org)
  model_low_res = Model(inputs=input_layer_org, outputs=output_layer_low)

  # Apply the downsampling transformation
  x_train_low = model_low_res.predict(x_train)
  x_test_low = model_low_res.predict(x_test)

  # Adjust the input shape for the downsampled image
  if k.image_data_format() == 'channels_first':
      input_shape_low_res = (1, 7, 7)
  else:
      input_shape_low_res = (7, 7, 1)

  return x_train_low, x_test_low, input_shape_low_res


def build_low_res(input_x_low_res):
  input_x_low = Input(shape=input_x_low_res)
  layer1 = Conv2D(16, (3, 3), activation='relu')(input_x_low)
  layer2 = MaxPooling2D((2, 2))(layer1)
  layer3 = Conv2D(32, (2, 2), activation='relu')(layer2)
  layer4 = Flatten()(layer3)
  layer5 = Dense(32, activation='relu')(layer4)
  output_low = Dense(10, activation='softmax')(layer5)

  model_low = Model(inputs=[input_x_low], outputs=output_low)
  model_low.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
  return model_low


def train_fit(model, x_train, y_train,x_test, y_test):
  # Train the model
  results = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

  model.summary()
  test_loss, test_acc = model.evaluate(x_test, y_test)
  print('Test loss:', test_loss)
  print('Test accuracy:', test_acc)