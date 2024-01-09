"""
This program calculates the accuracy of the WiSARD weightless 
model using the Threshold Adaptive method for image binarization.

Author: Luis Armando Quintanilla Villon
Date: July/2021    
"""


import wisardpkg as wp  # pip install wisardpkg==2.0.0a7
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cv2 as cv
import csv


def binarize_list(image):
    image2 = cv.GaussianBlur(image, (9, 9), 0)
    th = cv.adaptiveThreshold(image2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 17, 2)
    aux = np.where(th > 0, 1, 0).tolist()
    return [n for ele in aux for n in ele]


def binary_encoder_list(images_list):
    aux = [binarize_list(image) for image in images_list]
    return aux


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

## Preparing and transforming data
train_label = train_label.astype(str)
train_label = [str(n) for n in train_label]
test_label = [str(n) for n in test_label]
number_testes = 60000
train_bin_list = binary_encoder_list(train_data[:number_testes])
test_bin_list = binary_encoder_list(test_data[:10000])


# --------------------------------------- TEST -------------------------------------
# model = wp.Wisard(20)  # represents n-tuple size
# model.train(train_bin_list, train_label[:number_testes])  # .tolist())
# result = model.classify(test_bin_list)
# # print(result)
# print(accuracy_score(test_label[:10000], result))  # Result of classification


## Calsulating and saving the results
print("Threshold Adaptive")
with open('results/kmnist-threshold-adaptive.csv', 'w+') as file:
    writer = csv.writer(file)
    writer.writerow(["Tuple", "Accuracy"])
    for x in range(3, 43):
        model = wp.Wisard(x)  # represents n-tuple size
        model.train(train_bin_list, train_label[:number_testes])  # .tolist())
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



