import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

plt.gray() 

plt.matshow(digits.images[100])

plt.show()
plt.clf()
#print(digits.target[100])

model = KMeans(n_clusters=10, random_state=10)

model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()
plt.clf()

new_samples = np.array([
[0.00,0.00,0.15,2.14,3.35,3.81,0.76,0.00,1.07,3.96,7.09,7.62,7.62,6.94,1.83,0.00,7.47,7.63,4.80,2.29,1.07,0.76,0.31,0.00,7.55,7.62,7.62,7.62,7.62,7.62,7.09,1.68,2.21,3.05,3.05,3.05,2.29,2.59,7.09,4.73,0.00,0.00,0.00,0.00,0.00,0.00,5.41,6.40,1.75,1.37,0.00,0.00,1.30,3.05,6.10,7.62,5.87,6.86,3.81,4.27,7.17,7.62,7.40,3.20],
[0.00,4.88,3.28,0.00,1.14,7.62,5.49,0.00,0.00,6.86,4.57,0.00,0.00,4.65,7.09,0.00,0.00,6.86,5.18,1.52,2.13,4.88,7.62,3.35,0.00,6.71,7.62,7.62,7.62,7.62,7.62,7.55,0.00,0.92,1.52,1.52,1.37,6.33,5.64,0.00,0.00,0.00,0.00,0.00,0.46,7.62,3.58,0.00,0.00,0.00,0.00,0.00,0.76,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.76,7.62,3.51,0.00],
[0.38,2.29,2.67,3.05,3.58,3.81,3.81,2.67,4.42,7.62,7.62,7.62,7.55,7.40,7.62,7.40,0.92,1.52,0.76,0.08,2.74,7.40,6.48,2.14,0.00,0.00,0.00,5.11,7.62,6.18,0.53,0.00,0.00,0.00,1.30,7.62,4.80,0.15,0.00,0.00,0.00,0.15,6.18,6.94,0.84,0.00,0.00,0.00,0.00,4.35,7.62,2.44,0.00,0.00,0.00,0.00,0.00,7.40,5.03,0.08,0.00,0.00,0.00,0.00],
[0.00,1.83,6.86,7.62,7.62,7.32,1.07,0.00,0.00,3.36,7.55,2.97,2.29,7.62,2.29,0.00,0.00,0.08,1.22,0.00,1.52,7.62,2.29,0.00,0.00,0.00,0.00,0.00,3.28,7.62,1.83,0.00,0.00,0.00,0.00,3.58,7.62,6.33,0.15,0.00,0.00,0.00,2.90,7.47,6.86,1.83,1.52,1.07,0.00,1.83,7.47,7.62,7.62,7.62,7.62,7.24,0.00,1.07,4.50,3.05,1.52,1.52,1.52,1.07]
])

new_labels = model.predict(new_samples)
print(new_labels)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
