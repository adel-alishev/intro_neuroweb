import matplotlib.pyplot as plt
from matplotlib import image

img = image.imread('adel.jpg')
img = plt.imshow(img)
plt.show()