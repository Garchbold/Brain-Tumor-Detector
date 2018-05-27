import scipy.io
import numpy as np

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

#dataset
digits = datasets.load_digits()

print("Digits: ", digits)
print("Digits Data: ", digits.data) #an array of 1797 arrays, each of these arrays inside the main array have size 64 (8x8 represenation of images)
print("Digits Data type: ", type(digits.data))
print("Digits Data list len: ", len(digits.data.tolist()[0]))
print()
print("Digits Data Shape (# of rows, # of cols): ", digits.data.shape) #digits.data.shape represents (# of samples/images, dimemsionality i.e. 8x8 matrix of integers)
print("Digits Data Shape type: ", type(digits.data.shape))
print()
#print("Digits Images: ", digits.images) #digits.images is an array of multiple 2d arrays representing all 1797 images; each 2d array is 8x8 of integers
print("Digits Images type: ", type(digits.images))
print("Digits Images # of images: ", len(digits.images))
print("Digits Images # of rows: ", len(digits.images[0]))
print("Digits Images # of cols: ", len(digits.images[0][0]))
print()
print("Digits Target: ", digits.target)
print("Digits Target type: ", type(digits.target))
#print("Digits Target list: ", digits.target.tolist())
print("Digits Target length: ", len(digits.target.tolist())) #digits.target is an ndarray of integers, each integer corresponding to what each image should represent (e.g. the first image should be 0)


# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

images_and_labels = list(zip(digits.images, digits.target))
#print(images_and_labels) #creates a list of objects of the following form: 8x8 arrays with TARGET label number 8
    #ONE array in the main list of 1797 arrays is below:
'''(array([[ 0.,  0., 10., 14.,  8.,  1.,  0.,  0.],
			   [ 0.,  2., 16., 14.,  6.,  1.,  0.,  0.],
			   [ 0.,  0., 15., 15.,  8., 15.,  0.,  0.],
			   [ 0.,  0.,  5., 16., 16., 10.,  0.,  0.],
			   [ 0.,  0., 12., 15., 15., 12.,  0.,  0.],
			   [ 0.,  4., 16.,  6.,  4., 16.,  6.,  0.],
			   [ 0.,  8., 16., 10.,  8., 16.,  8.,  0.],
			   [ 0.,  1.,  8., 12., 14., 12.,  1.,  0.]]), 8)'''

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) #1797 x 64 ndarray
print(data)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001) #0.001 is a good value; any higher or lower gives worse results
print(classifier)

# Model learns based on first half of data
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2]) #// is floor division
print(data[:n_samples // 2][0]) #first half of the image data ~898 x 64
print(digits.target[:n_samples // 2]) #first half of the target data ~898

print()
print("....Begin prediction on 2nd half of data.....")
# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:] #second half of the target data
predicted = classifier.predict(data[n_samples // 2:]) #second half of image data
print("Expected nums: ", expected)
print("Predicted nums: ", predicted)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

print("Accuracy of model: ", classifier.score(data, digits.target))
plt.show() #comment this to avoid opening a graph









