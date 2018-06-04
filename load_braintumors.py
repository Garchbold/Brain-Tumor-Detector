import scipy.io
import numpy as np
import os
import errno
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from mat_convert import setup_data

def load_braintumors(n_class=3, return_X_y=False):
    """Load and return the digits dataset (classification).
    Each datapoint is a 8x8 image of a digit.
    =================   ==============
    Classes                         2
    Samples per class               UNKNOWN (TODO)
    Samples total                   767
    Dimensionality                  262144
    Features                        (TODO)
    =================   ==============
    Read more in the :ref:`User Guide <datasets>`.
    Parameters
    ----------
    n_class : integer, between 0 and 1, optional (default=1)
        The number of classes to return.
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
        .. versionadded:: 0.18
    """

    #print(data) #~700 rows, 512*512 + 1 columns
    #data: each row represents an image encoded in the following way: 
    #   512*512 + 1 columns of integers per row:
    #       the first 512*512 columns should be split into 512 arrays of 512 integers; the last column is TARGET number (what integer each image actually represents and this should be predicted correctly)

    data = setup_data()

    print("\nBegin loading brain tumors OLD...\n")

    descr = "Brain Tumor Image Classifier"
    #print("Data: ", data)

    target = data[:, -1].astype(np.int) #ndarray type, contains int64's; for all rows, slice and take only the last column which is the target data
    #print("Target: ", target)
    #new_target = np.random.permutation(target)
    #print("New target: ", new_target)

    data_without_target = data[:, :-1] #ndarray type; for all rows, slice and take all columns EXCEPT the last column (the target data)
    #print("data_without_target: ", data_without_target)
    #print("Data no target length: ", len(data_without_target[0]))
    #print("length data: ", len(data[0]))
    
    images = data_without_target.view() #same as data_without_target....
    #print("images: ", images)

    #images.shape at this point is equal to (1797, 64)
    images.shape = (-1, 512, 512) #now images.shape is (num_images, 512, 512)
    print("(# of images, dim_x, dim_y: ", images.shape)

    if n_class < 3: #filter out any classes that shouldn't be considered
        print("n_class is < 3")
        #print(target)
        filter_out_classes = target < n_class
        #print("filter_out_classes: ", filter_out_classes)

        data_without_target, target = data_without_target[filter_out_classes], target[filter_out_classes]
        #print("data_without_target: ", data_without_target)

        #print("target: ", target)

        images = images[filter_out_classes]
        #print("images[filter_out_classes]: ", images)

    if return_X_y: #returns a tuple: (image data without target data, target data)
        return data_without_target, target


#   data : Bunch
        #Dictionary-like object, the interesting attributes are:

        #'data' = the data to learn, 
        #'images' = the image data (8x8 of integers, an array of size 8, with 8 sub-arrays of size 8) corresponding to each image, 
        #'target' = the classification labels for each sample, i.e. what integers the images actually are supposed to be, 
        #'target_names' = the meaning of the labels; this is just an array of integers from 0 to 9 in this case,
        #'DESCR', the full description of the dataset.
    return Bunch(data=data_without_target,
                 target=target,
                 target_names=np.arange(1, 4),
                 images=images,
                 DESCR=descr)

#a = load_braintumors()
#print()
#print("Returned: ", a)