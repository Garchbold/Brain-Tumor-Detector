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
print("brains Data: ", brains.data) #an array of 1797 arrays, each of these arrays inside the main array have size 64 (8x8 represenation of images)
print("brains Data type: ", type(brains.data))
print("brains Data list len: ", len(brains.data.tolist()[0]))
print()
print("brains Data Shape (# of rows, # of cols): ", brains.data.shape) #brains.data.shape represents (# of samples/images, dimemsionality i.e. 8x8 matrix of integers)
print("brains Data Shape type: ", type(brains.data.shape))
print()
#print("brains Images: ", brains.images) #brains.images is an array of multiple 2d arrays representing all 1797 images; each 2d array is 8x8 of integers
print("brains Images type: ", type(brains.images))
print("brains Images # of images: ", len(brains.images))
print("brains Images # of rows: ", len(brains.images[0]))
print("brains Images # of cols: ", len(brains.images[0][0]))
print()
print("brains Target: ", brains.target)
print("brains Target type: ", type(brains.target))
#print("brains Target list: ", brains.target.tolist())
print("brains Target length: ", len(brains.target.tolist())) #brains.target is an ndarray of integers, each integer corresponding to what each image should represent (e.g. the first image should be 0)



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
print(data)
print("Data: ", data)

# create a linear SVC
linear_classifier = svm.LinearSVC()

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
#plt.show() #comment this to avoid opening a graph






