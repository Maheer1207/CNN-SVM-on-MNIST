from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import numpy as np
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten
from keras.models import Model
import matplotlib.pyplot as plt

def svm_mnist(res):
  x_train, y_train, x_test, y_test = svm_load()

  if(res=="low"):
    x_train, x_test = svm_down_sample(x_train, x_test)

  x_train, x_test = svm_process(x_train, x_test)

  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

  svm_model = svm_build()
  svm_train_fit(x_train, y_train, x_val, y_val, svm_model)

  return svm_model, x_train, y_train, x_test, y_test

def svm_load():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  return x_train, y_train, x_test, y_test

def svm_process(x_train, x_test):
  x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255
  x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255
  return x_train, x_test

def svm_build():
  svm_model = SVC(kernel='rbf', gamma='scale', max_iter=250)
  return svm_model

def svm_train_fit(x_train, y_train, x_test, y_test, svm_model):
  svm_model.fit(x_train, y_train)
  test_predictions = svm_model.predict(x_test)

  # Evaluate the model
  accuracy = accuracy_score(y_test, test_predictions)
  print(f"Test Accuracy: {accuracy}")


def svm_down_sample(x_train, x_test):
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)

  input_x = (28, 28, 1)  # Original shape for MNIST
  input_layer_org = Input(shape=input_x)
  output_layer_low = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer_org)
  model_low_res = Model(inputs=input_layer_org, outputs=output_layer_low)

  x_train_low = model_low_res.predict(x_train)
  x_test_low = model_low_res.predict(x_test)

  # Reshape back to fit SVM input requirements
  x_train_low = x_train_low.reshape((x_train_low.shape[0], -1))
  x_test_low = x_test_low.reshape((x_test_low.shape[0], -1))

  return x_train_low, x_test_low