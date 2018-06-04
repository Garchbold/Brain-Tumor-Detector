# Brain-Tumor-Detector

### Classification of 3 different types of brain tumors using an MRI dataset and SciKit
**Classification is accomplished via Support Vector Machines (supervised learning).**

##### We classify images using three different kinds of methods:
1. SVC (C-Support Vector Classification)
2. LinearSVC (Linear Support Vector Classification)
3. NuSVC (Nu-Support Vector Classification)

##### We label tumors with the following integers: 
- 1=meningioma
- 2=glioma
- 3=pituitary tumor


The image dataset can be found in **braintumors_1/**

<br>

#### Usage: `python3 classifer_TYPE.py`
- SVC: `classifer_SVC.py`
- LinearSVC: `classifer_linearSVC.py`
- NuSVC: `classifer_NuSVC.py`


*Python Dependencies:* `scipy.io`, `numpy`, `matplotlib.pyplot`, `sklearn`, `h5py`, `os`, `errno`, `random`

<br>
