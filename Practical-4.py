import cv2
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
# Replace with your image name
image = cv2.imread('images.jpg')
  # Convert BGR to RGB for correct display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Display original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
   print("Number of keypoints detected:", len(keypoints))
 sift_image = cv2.drawKeypoints(
    image_rgb,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
  plt.figure(figsize=(10, 6))
plt.imshow(sift_image)
plt.title("SIFT Keypoints")
plt.axis('off')
