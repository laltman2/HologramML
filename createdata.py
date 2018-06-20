import numpy as np
from spheredhm import spheredhm
import json, base64
import matplotlib.pyplot as plt

#values that won't change
npix = 201
dim = [npix, npix]
mpp = 0.135
n_m = 1.339
lamb = 0.447

def createdata(n_samp): 
    Xset = []
    Yset = []

    #x = np.random.uniform(low= -npix/3, high = npix/3)
    #y = np.random.uniform(low= -npix/3, high = npix/3)
    for i in range(0, n_samp):
        z = np.random.uniform(low= 50, high = 600)
        z = z/mpp
        rp = [0,0,z]
        #a_p = np.random.uniform(low= 0.1, high = 1)
        #n_p = np.random.uniform(low=0.1, high=3)
        image = spheredhm(rp, 0.5, 1.5, n_m , dim, lamb = lamb, mpp = mpp)
        Xset.append(image)
        Yset.append(z)
        # Plot the hologram.
        #plt.imshow(image)
        #plt.title("Test Hologram")
        #plt.gray()
        #plt.text(0,0,z)
       # plt.show()
    return [Xset, Yset]


# Plot the hologram.
#plt.imshow(image)
#plt.title("Test Hologram")
#plt.gray()
#plt.show()

#print(rp, a_p, n_p)

#data2 = createdata(2)
#print(data2[0])
#data2[0][0] = data2[0][0].tolist()
#data2[0][1] = data2[0][1].tolist()
#data = {'training': [{'train_x': data2[0], 'train_y': data2[1]}]}
#json.dump(data, open('test.txt', 'w'))


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
