import numpy
try:
    import h5py
except ImportError:
    h5py = None
from keras.models import load_model
#from keras import load_weights
import json
from keras import backend as K
from matplotlib import pyplot as plt

with open('data.txt', 'r') as file:
    data = json.load(file)
x_train = numpy.array(data["training"][0]["train_x"])
y_train = numpy.array(data["training"][0]["train_y"])
x_test = numpy.array(data["testing"][0]["test_x"])
y_test = numpy.array(data["testing"][0]["test_y"])

#format data

img_rows = 201
img_cols = 201
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train *= 100./255
x_test *= 100./255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

#print(x_test[0][:][0].shape)

#Add gaussian noise to data                                                                                            
def noisy(img):
    row,col,ch=input_shape
    mean=0
    var=10**(-5)
    sigma=var**0.5
    gauss=numpy.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    return noisy

print('Adding noise...')
for img in x_train:
    img = noisy(img)

for img in x_test:
    img = noisy(img)

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#json_file = open('model_arch.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()

model = load_model('newmodel.h5')
#model.load_weights('model_weights.h5', by_name=True)

y_pred = model.predict(x_train)
#print(numpy.mean(y_train))

plt.plot(y_train, y_pred, 'bo')
plt.show()
