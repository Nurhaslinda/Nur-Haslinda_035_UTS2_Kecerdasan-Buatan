#Nama   : Nur Haslinda
#NIM    : 21091397035
#Kelas  : 2021 A
#Multi Neuron Batch Input 2 layer

#inisialisasi numpy
import numpy as np

#inisialisasi variabel dengan matriks 6x10 (input 10 dan batch 6)
inputs = [[0.1, 1.0, 0.2, 2.0, 0.3, 3.0, 0.4, 4.0, 0.5, 5.0],
          [0.6, 7.0, 0.8, 0.9, 6.0, 7.0, 8.0, 9.0, 2.9, 2.9],
          [2.5, 3.1, 4.0, 8.0, 2.4, 2.9, 0.3, 0.4, 1.7, 1.8],
          [0.3, 1.7, 1.8, 1.9, 2.7, 2.0, 7.0, 4.0, 0.8, 2.8],
          [0.2, 0.9, 1.9, 1.0, 1.1, 1.2, 0.7, 0.3, 0.6, 1.7],
          [0.8, 0.3, 0.7, 0.2, 0.1, 1.7, 8.0, 3.0, 4.0, 9.0]]

#bobot per neuron layer 1
#panjang weights layer 1 sesuai dengan panjang input, dan jumlah weights sesuai dengan jumlah neuron layer 1 (10x5)
weights1 = [[1.0, 3.0, 0.3, 2.1, 1.2, 1.9, 4.0, 9.0, 1.1, 1.7],
           [5.0, 8.0, 1.0, 6.0, 7.0, 0.1, 2.0, 3.0, 4.0, 2.9],
           [9.0, 6.0, 8.0, 0.5, 1.9, 1.7, 0.2, 2.6, 2.4, 1.8],
           [3.0, 6.0, 1.2, 1.8, 2.4, 3.0, 2.0, 4.0, 8.0, 1.6],
           [1.6, 1.0, 2.2, 1.7, 0.3, 1.9, 2.7, 0.2, 1.8, 3.0]]

#bias per neuron layer 1
#jumlah bias layer 1 sesuai dengan jumlah neuron layer 1
biases = [2.0, 3.0, 0.5, 2.4, 2.9]

#bobot per neuron layer 2
#panjang weights layer 2 sesuai dengan panjang outputs layer 1 dan jumlah weights sesuai dengan jumlah neuron layer 2 (5x3)
weights2 = [[2.5, 2.4, 0.1, 4.8, 2.9,],
            [3.6, 2.5, 7.0, 1.0, 6.0],
            [0.8, 0.4, 0.3, 0.2, 0.9]]

#bias per neuron layer 2
#Jumlah bias layer 2 sesuai dengan jumlah neuron layer 2
biases2 = [4.0, 9.0, 1.7]

#ouputs dengan menggunakan metode numpy
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases
layer2_outputs = np.dot (layer1_outputs, np.array(weights2).T) + biases2

#print ouputs
print(layer2_outputs)
