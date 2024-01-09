"""
This program calculates the accuracy of the WiSARD weightless 
model using a simple Threshold method for image binarization.

Author: Luis Armando Quintanilla Villon
Date: July/2021    
"""


import wisardpkg as wp
import numpy as np
from sklearn.metrics import accuracy_score
import csv


def binarize(image, threshold):
    return np.where(image > threshold, 1, 0).tolist()


def binary_encoder(images, threshold=1):
    return [binarize(image, threshold) for image in images]


def binarize_list(image, threshold):
    aux = np.where(image > threshold, 1, 0).tolist()
    return [n for ele in aux for n in ele]


def binary_encoder_list(images, threshold=0):
    return [binarize_list(image, threshold) for image in images]


## Reading dataset
DATA_PATH = 'dataset'
train_data = np.load(DATA_PATH + '/kmnist-train-imgs.npz')['arr_0']
train_label = np.load(DATA_PATH + '/kmnist-train-labels.npz')['arr_0']
test_data = np.load(DATA_PATH + '/kmnist-test-imgs.npz')['arr_0']
test_label = np.load(DATA_PATH + '/kmnist-test-labels.npz')['arr_0']

# ## Shape information
# print("train_data: {}".format(train_data.shape))
# print("train_label: {}".format(train_label.shape))
# print("test_data: {}".format(test_data.shape))
# print("test_label: {}".format(test_label.shape))

# print(train_data[0])
train_label = [str(n) for n in train_label]
test_label = [str(n) for n in test_label]
train_bin = binary_encoder(train_data)
train_bin_list = binary_encoder_list(train_data)
test_bin = binary_encoder(test_data)
test_bin_list = binary_encoder_list(test_data)

# # --------------------------------------- TEST -------------------------------------
# model = wp.Wisard(20)  # represents n-tuple size
# model.train(train_bin_list, train_label)  # .tolist())
# result = model.classify(test_bin_list)
# # print(result)
# print(accuracy_score(test_label[:10000], result)*100)  # Result of classification

## Calsulating and saving the results
print("Simple threshold")
with open('results/kmnist-threshold-simple.csv', 'w+') as file:
    writer = csv.writer(file)
    writer.writerow(["Tuple", "Accuracy"])
    for x in range(3, 43):
        model = wp.Wisard(x)  # represents n-tuple size
        model.train(train_bin_list, train_label)  # .tolist())
        result = model.classify(test_bin_list)
        accuracy = accuracy_score(test_label, result)*100
        writer.writerow([x,accuracy])
        print(accuracy)

# # # # Use of the ClusWiSARD model
# for x in range(3, 49):
#     addressSize = x  # number of addressing bits in the ram.
#     minScore = 0.5  # min score of training process
#     threshold = 50  # limit of training cycles by discriminator
#     discriminatorLimit = 2  # limit of discriminators by clusters
#     # False by default for performance reasons
#     # when enabled,e ClusWiSARD prints the progress of train() and classify()
#     verbose = False
#     model_clus = wp.ClusWisard(addressSize, minScore, threshold, discriminatorLimit, verbose=False)
#     model_clus.train(train_bin_list, train_label[:number_testes])  # .tolist())
#     result = model_clus.classify(test_bin_list)
#     # print(result)
#     print(accuracy_score(test_label[:10000], result)*100)