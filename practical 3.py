import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

image_path = "Elephant.jpg" 

image = cv2.imread(f"/content/Elephant.jpg") 

if image is None:
    raise ValueError("Image not found. Check the image path.")

height, width = image.shape[:2]
new_width = 400
new_height = int(new_width * (height / width))
image_resized = cv2.resize(image, (new_width, new_height))

image_rgb_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) 

gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

features, hog_image = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb_original)
plt.axis("off")

plt.subplot(1, 2, 2) 
plt.title("HOG Feature Visualization")
plt.imshow(hog_image_rescaled, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

print("Original image shape:", image.shape)
print("Resized image shape:", image_resized.shape)
print("HOG image shape:", hog_image.shape)
print("HOG feature vector length:", len(features))
