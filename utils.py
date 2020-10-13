import numpy as np 
import cv2
import os
import time

import tensorflow
from tensorflow.keras.layers import Input , Conv2D , Dense , Flatten , MaxPooling2D
from tensorflow.keras import Model

def load_model(input_shape , debug=False):
	input = Input(input_shape)
	x = Conv2D(filters = 5 , kernel_size = (2,2) , padding="valid", activation = 'relu')(input)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(filters = 8 , kernel_size = (4,4) , padding="valid", activation = 'relu')(x)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(filters = 8 , kernel_size = (2,2) , padding="valid", activation = 'relu')(x)
	x = Conv2D(filters = 10 , kernel_size = (2,2) , padding="valid", activation = 'relu')(x)
	x = Flatten()(x)
	x = Dense(500 , activation = 'relu')(x)
	x = Dense(100 , activation = 'relu')(x)
	x = Dense(50 , activation = 'relu')(x)
	x = Dense(3 , activation = 'softmax')(x)

	model = Model(inputs = input , outputs = x)
	if debug: model.summary()
	return model

def crop (image , size ,debug=False):
  width , hight,_ = image.shape
  if debug: print('width={} , hight={}'.format(width , hight))
  target_width , target_hight = size
  center_x = int(width/2)
  center_y = int(hight/2)
  if debug: print('x={} , y={}'.format(center_x , center_y))
  img = image[center_x - int(target_width/2) : center_x + int(target_width/2) , center_y - int(target_hight/2) : center_y + int(target_hight/2) , :]
  if debug: print(img.shape)
  return img

def predict(model , image , debug=False):
  if(image.shape[0] > 1500 and image.shape[1]>1500):
    image = crop(image=image , size=(1500,1500))
  image = image/255.0
  image = cv2.resize(image , (120,120))
  image = np.expand_dims(image , axis=0)
  start = time.time()
  pred = model.predict(image)
  end = time.time()
  if debug: print("Prediction time = {} seconds".format(end-start))
  if(any(item > 0.50 for item in pred[0])):
    ind = np.argmax(pred)
    out = np.zeros(len(pred[0]))
    out[ind] = 1
    confidence = 1
  else:
    out = np.zeros(len(pred[0]))
    confidence = 0

  if debug:
    for i in range(len(pred[0])):
      print("Class {} score: {}".format(i+1 , pred[0,i]*100))

  return confidence,out


def translate(arr):
	if np.argmax(a=arr) == 0:
		return 'paper'
	elif np.argmax(a=arr) == 1:
		return 'rock'
	elif np.argmax(a=arr) == 2:
		return 'scissors'

