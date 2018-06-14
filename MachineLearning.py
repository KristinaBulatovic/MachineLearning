"""
version of Python 2 or Python 3

pip install numpy
pip install scipy
pip install matplotlib
pip install scikit-learn
"""

import matplotlib.pyplot as plt      # pyplot is used to actually plot a chart
from sklearn import datasets         # datasets are used as a sample dataset, which contains one set that has number recognition data
from sklearn import svm              # import svm - which is for the sklearn Support Vector Machine

digits = datasets.load_digits()      # we're defining the digits variable, which is the loaded digit dataset

# print(digits.data)                 # digits.data is the actual data (features)
# print(digits.target)               # digits.target is the actual label we've assigned to the digits data

clf = svm.SVC(gamma=0.0001, C=100)   # we specify the classifier and we set gamma and C

# print(len(digits.data))

x,y = digits.data[:-10], digits.target[:-10]   # assign the value into x and y
# This loads in all but the last 10 data points, so we can use all of these for training.
# Then, we can use the last 10 data points for testing.
# The X contains all of the "coordinates" and y is simply the "target" or "classification" of the data.
# Each bit of data pertains to a number. So X may contain a bunch of pixel data for the number 5,
# and the "y" would be 5.


clf.fit(x,y)                        # we train with clf.fit(x,y)

print('Prediction:',clf.predict(digits.data[[-2]]))          # test it!
# This will predict what the 5th from the last element is.

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# this just shows us an image of the number in question.


