import scipy.io
import numpy as np

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import classifiers and performance metrics
from sklearn import svm, metrics

from load_braintumors import load_braintumors

#dataset
brains = load_braintumors()

print("brains: ", brains)

images_and_labels = list(zip(brains.images, brains.target))
#print(images_and_labels) #creates a list of objects of the following form: 512x512 arrays with TARGET label number

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


# reshape data
n_samples = len(brains.images)
data = brains.images.reshape((n_samples, -1)) # num_samples x 512*512 ndarray
print("Data: ", data)

# create a linear SVC
linear_classifier = svm.LinearSVC()
print(linear_classifier)

#print("Second half brains: ", brains.target[:n_samples // 2])
#print(np.unique(brains.target[:n_samples // 2], return_inverse=True)) # initially failed in linear_classifier.fit below

# learn based on first half of data
linear_classifier.fit(data[:n_samples // 2], brains.target[:n_samples // 2]) #// is floor division
print("First half of data: ", data[:n_samples // 2][0]) #first half of the image data ~898 x 64
print("First half of targets: ", brains.target[:n_samples // 2]) #first half of the target data ~898

unique, counts = np.unique(brains.target[:n_samples // 2], return_counts=True)
print(dict(zip(unique, counts)))

print()
print("....LINEAR SVC: Begin prediction on 2nd half of data.....")
# predict based on the second half of the data:
expected = brains.target[n_samples // 2:] #second half of the target data
predicted = linear_classifier.predict(data[n_samples // 2:]) #second half of image data
print("Expected nums: ", expected)
print("Predicted nums: ", predicted)

unique2, counts2 = np.unique(predicted, return_counts=True)
print("Predicted nums (counts): ", dict(zip(unique2, counts2)))

print("Classification report for linear classifier %s:\n%s\n"
      % (linear_classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(brains.images[n_samples // 2:], predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

print("Accuracy of model: ", linear_classifier.score(data, brains.target))
plt.show() #comment this to avoid opening a graph

