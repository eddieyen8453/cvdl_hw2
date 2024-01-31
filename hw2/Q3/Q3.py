import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


# Step 1: Convert RGB image to grayscale
def rgb_to_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Step 2: Normalize grayscale image to [0, 1]
def normalize_image(image):
    normalized_image = image / 255.0
    return normalized_image

# Step 3: Use PCA for dimensionality reduction
def apply_pca(image, n_components):
    h, w = image.shape
    flattened_image = image.flatten().reshape(-1, 1)

    pca = PCA(n_components=n_components)
    reduced_image = pca.fit_transform(flattened_image)
    reconstructed_image = pca.inverse_transform(reduced_image).reshape(h, w)

    return reconstructed_image

# Step 4: Use MSE to compute reconstruction error
def compute_mse(original, reconstructed):
    mse = mean_squared_error(original, reconstructed)
    return mse

# Step 5: Find minimum n with MSE less than or equal to 3.0
def find_min_components(original, max_components, threshold_error):
    for n_components in range(1, max_components + 1):
        reconstructed_image = apply_pca(original, n_components)
        mse = compute_mse(original, reconstructed_image)

        print(f"Components: {n_components}, MSE: {mse}")

        if mse <= threshold_error:
            return n_components

    return max_components

from sklearn.metrics import mean_squared_error

class Question3:
    def __init__(self):

        self.originalImageGrayList = []
        self.normalisedImageGrayList = []
        self.IMG_HEIGHT = None
        self.IMG_WIDTH = None



    def imageReconstruction(self, dirPath):
        
        #Step 1
        img = cv2.imread(dirPath,0)

        #Step 2
        normalisedImage = normalize_image(img)

        #Step 3
        min_components = min(img.shape)
        max_components = 1000
        threshold_error = 3.0

        optimal_components = find_min_components(normalisedImage, max_components, threshold_error)
        
        # Reconstruct the image using the optimal number of components
        reconstructed_image = apply_pca(normalisedImage, optimal_components)
        
        plt.subplot(1, 2, 1)
        plt.title('Gray Scale Image')
        plt.imshow(normalisedImage, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title(f'Reconstructed Image (n={optimal_components})')
        plt.imshow(reconstructed_image, cmap='gray')

        plt.show()

        

        



if __name__ == "__main__":
    # for testing
    Q3 = Question3()
    Q3.imageReconstruction()