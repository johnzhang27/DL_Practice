import config
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = config.PATH
CATEGORIES = ["train","val"]
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break
print(img_array)# img_shape could give you the size of the matrix.
# All img in my data set is 512 x 256 so I dont need to resize them to same size.
# cv2.resize(img_array,(#,#))
CATEGORIES = ["train", "val"]
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(50,50))
                training_data.append([new_array,class_num])
            except Exception as e:
                print("img broken")

create_training_data()
print(len(training_data))

X = []# features
y = []
#y = [] y is label, but in this case I did not use a label in training data so I could skip this part.
for features,label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1,50,50,1) # If color image then change the last 1 to 3.
# numPy array is a multi-dimensional array and matrix data structure.
'''import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)'''
# This np save method is one of the alternative way from
# comment, change it back if not working.
np.save('features.npy', X)
X = np.load('features.npy')
np.save('label.npy', y)
y = np.load('label.npy')
print(X[1])