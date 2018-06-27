import numpy as np
import sys
#or wherever you have theory stored
sys.path.append('/Users/laurenaltman/Desktop/Summer_Research/lorenzmie/theory')
from spheredhm import spheredhm
import json, base64
import matplotlib.pyplot as plt
from keras import backend as K

#values that won't change
npix = 501
dim = [npix, npix]
mpp = 0.135
n_m = 1.339
lamb = 0.447

#Add gaussian noise to data                                                
def noisy(img):
    row,col,ch= npix, npix, 1
    mean=0
    var=1e-06
    sigma=var**0.5
    gauss=np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    return noisy

def format_image(img):
    if K.image_data_format() == 'channels_first':
        img = img.reshape(1, npix, npix)
    else:
        img = img.reshape(npix, npix, 1)
    return img

 
def createdata(n_samp): 
    Xset = []
    Yset = []
    #x = np.random.uniform(low= -npix/3, high = npix/3)
    #y = np.random.uniform(low= -npix/3, high = npix/3)
    for i in range(0, n_samp):
        z = np.random.uniform(low= 50, high = 600)
        z = z/mpp
        rp = [0,0,z]
        a_p = np.random.uniform(low= 0.2, high = 5.)
        n_p = np.random.uniform(low=1.38, high=2.5)
        image = spheredhm(rp, a_p, n_p, n_m , dim, lamb = lamb, mpp = mpp)
        image = format_image(image)
        image = noisy(image)
        Xset.append(image)
        Yset.append(z)
    return [Xset, Yset]


# Plot the hologram.
#plt.imshow(image)
#plt.title("Test Hologram")
#plt.gray()
#plt.show()

#print(rp, a_p, n_p)

[testimg, testz] = createdata(1)
lum_img = testimg[0][:,:,0]
plt.imshow(lum_img)
plt.gray()
plt.show()
print(testz)

if __name__ == '__main__':
    train_num = 100
    test_num = 10
    training_set = createdata(train_num)
    for i in range(0,train_num):
        training_set[0][i] = training_set[0][i].tolist()
    test_set = createdata(test_num)
    for i in range(0,test_num):
        test_set[0][i] = test_set[0][i].tolist()
    data = {'training': [{'train_x': training_set[0], 'train_y': training_set[1]}],'testing': [{'test_x': test_set[0], 'test_y': test_set[1]}]}
    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)
        #json.dump(training_set[0], outfile)
