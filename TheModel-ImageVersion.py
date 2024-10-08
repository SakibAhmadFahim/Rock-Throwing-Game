import numpy

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.activations import relu, linear


imageResize = (80,30)

# print('loading image data...')
img_data = numpy.load('modelImageData.npy', allow_pickle=True)
# print('loaded!!')

x = numpy.array([i for i in img_data[0]])
y = numpy.array([i for i in img_data[1]])

# The code extracts the input data x and output data y from img_data, and scales the input data by dividing it by 255 to normalize the pixel values.
x = x / 255

# def myLoss(y_pred, y_ac):
#     d = y_pred - y_ac
#     sum = d[0] * d[0] + d[1] * d[1]
#     return 2*sum

# the model

model = Sequential()
# specifies the number of filters/channels in the layer. 
model.add(Conv2D(filters=16, kernel_size=(3, 3), kernel_initializer='normal', activation=relu, input_shape=x[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))   
#  size of the pooling window. 

model.add(Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='normal', activation=relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='normal', activation=relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

# converts the multi-dimensional arrays into flattened one-dimensional arrays or single-dimensional arrays
model.add(Flatten())

model.add(Dense(units=64, kernel_initializer='normal', activation=relu))

model.add(Dense(units=2, kernel_initializer='normal', activation=linear))


model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.fit(x, y, epochs=10 , validation_split=0.1, shuffle=False)

# print the differences between the predicted values and actual values.
# def acc(s = 0):
#     for i in range(s, s+10):
#         a = model.predict(x[i].reshape((1,) + x[0].shape))
#         print(y[i] - a[0])

# acc()

model.save('model_imageToPrediction')