# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
#git practice
#1st feature is color histogram
def color_histogram_vector(image, bins=(8, 8, 8)):
	# We are generating a 3-D colour histogram using HSV color space developed using the given bins for RGB space
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	color_histogram = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3
	else:
		cv2.normalize(hist, hist)
	# finally flattening the histogram color array to a 1-d array
	return hist.flatten()

#2nd feature is raw pixel intensities
def pixel_intensity_vector(image, size=(32, 32)):
	# We resized images to a fixed size and converted the image array to a 1-d flattened array
	return cv2.resize(image, size).flatten()

# constructing the argument parse and parsing the arguments
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
argp.add_argument("-k", "--neighbors", type=int, default=1, help="number of nearest neighbors for classification")
argp.add_argument("-j", "--jobs", type=int, default=-1, help="numbers of jobs(parallel processing) for k-NN distance(-1 uses all available cores)")
arguments = vars(argp.parse_args())

# Taking the image dataset
print("Loading the Images from the database")
imagePaths = list(paths.list_images(arguments["dataset"]))

# initialize the raw pixel intensities matrix, the features matrix, and labels list
rawImages = []
features = []
labels = []

# looping over the input images
for (i, imagePath) in enumerate(imagePaths):
	# loading the image and extracting the class label
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	# extracting raw pixel intensity "features", followed by a color histogram
	pixels = pixel_intensity_vector(image)
	color_histogram = color_histogram_vector(image)

	# updating the raw images, features, and labels matrices
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

#Partioning the dataset into 3:1 for training and testing purpose
(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)	

# Modelling and testing kNN classifier for the color histograms
print("Finding accuracy with color histograms")
model = KNeighborsClassifier(n_neighbors=arguments["neighbors"], n_jobs=arguments["jobs"])
model.fit(trainFeat, trainLabels)
accuracy = model.score(testFeat, testLabels)
print("Accuracy with color histograms: {:.2f}%".format(accuracy * 100))
	
# Modelling and testing kNN classifier for the raw pixel intensities
print("Finding accuracy with raw pixel intensities")
model = KNeighborsClassifier(n_neighbors=arguments["neighbors"], n_jobs=arguments["jobs"])
model.fit(trainRI, trainRL)
accuracy = model.score(testRI, testRL)
print("Accuracy with Raw Pixel Intensities: {:.2f}%".format(accuracy * 100))

