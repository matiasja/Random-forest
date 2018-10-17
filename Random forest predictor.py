import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random
from sklearn import ensemble

#Load Data
digits = load_digits()

#Define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Split into sample and testing data
sample_index = random.sample(range(len(x)), int(len(x)/2)) #50-50
test_index =[i for i in range(len(x)) if i not in sample_index]

sample_images = [x[i] for i in sample_index]
test_images = [x[i] for i in test_index]

sample_target = [y[i] for i in sample_index]
valid_target = [y[i] for i in test_index]

#Fit model to sample data using random tree classifier
classifier = ensemble.RandomForestClassifier()
classifier.fit(sample_images, sample_target)

#Predict testing data
score=classifier.score(test_images, valid_target)

#Prediction rate
print('Random Tree Classifier:\n')
print('Prediction rate\t'+str(score))

#Show 24 examples of predictions
rand = random.randint(1,len(digits.images)-24)
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(10,8))
for index, (image, label) in enumerate(images_and_labels[rand:rand+24]):
	plt.subplot(4,6, index + 1)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	x = classifier.predict(image.reshape(1,-1))
	plt.title('label %i\n' % label + 'prediction %i' % x)
plt.show()