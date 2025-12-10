import os
os.chdir(os.path.dirname(__file__))
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# 1.Load Image
img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for plotting in matplotlib

# 2. Box Filters
# 5x5 box filter
kernel_5 = np.ones((5,5), dtype=np.float32)
kernel_5_norm = kernel_5 / 25
box_5_non = cv2.filter2D(img, -1, kernel_5)
box_5_norm = cv2.filter2D(img, -1, kernel_5_norm)

# 20x20 box filter
kernel_20 = np.ones((20,20), dtype=np.float32)
kernel_20_norm = kernel_20 / 400
box_20_non = cv2.filter2D(img, -1, kernel_20)
box_20_norm = cv2.filter2D(img, -1, kernel_20_norm)

# 3. Compute σ and Gaussian filter size
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sigma = gray.std()
filter_size = int(round(2 * math.pi * sigma))
if filter_size % 2 == 0: #make filter size odd
    filter_size += 1
print("σ =", sigma)
print("Gaussian kernel size =", filter_size)

# 4. Separable Gaussian Filter (OpenCV built-in)
gaussian = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)

# 5. Separable Normalized Gaussian Filter (manual)
ax = np.linspace(-(filter_size // 2), filter_size // 2, filter_size)
gauss_1d = np.exp(-(ax*2) / (2 * sigma*2))
gauss_1d = gauss_1d / gauss_1d.sum()  # normalization

# horizontal pass
temp = cv2.filter2D(img, -1, gauss_1d.reshape(1, -1))
# vertical pass
gaussian_normalized = cv2.filter2D(temp, -1, gauss_1d.reshape(-1, 1))

#  6.Plot All Results
titles = ['Original', 'Box 5x5 Non-Norm', 'Box 5x5 Norm', 'Box 20x20 Non-Norm', 'Box 20x20 Norm',
          'Gaussian Separable', 'Gaussian Normalized']
images = [img_rgb, 
          cv2.cvtColor(box_5_non, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(box_5_norm, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(box_20_non, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(box_20_norm, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(gaussian_normalized, cv2.COLOR_BGR2RGB)]
plt.figure(figsize=(15,8))
for i in range(len(images)):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
#  7.Save results 
cv2.imwrite("box_5_non.jpg", box_5_non)
cv2.imwrite("box_5_norm.jpg", box_5_norm)
cv2.imwrite("box_20_non.jpg", box_20_non)
cv2.imwrite("box_20_norm.jpg", box_20_norm)
cv2.imwrite("gaussian_separable.jpg", gaussian)
cv2.imwrite("gaussian_normalized.jpg", gaussian_normalized)