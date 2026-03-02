import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
from io import BytesIO

# Upload image
uploaded = files.upload()

# Read uploaded image and convert to grayscale
for filename in uploaded.keys():
    img = Image.open(BytesIO(uploaded[filename])).convert('L')
    image = np.array(img)

window_size = 50
threshold = 100 # brightness threshold
step_size = 20

detections = []

height, width = image.shape

# Sliding Window (Region Proposal)
for y in range(0, height - window_size + 1, step_size):
    for x in range(0, width - window_size + 1, step_size):

        window = image[y:y+window_size, x:x+window_size]
        feature = np.mean(window)  # Feature extraction

        # Simple classification
        if feature > threshold:
            detections.append((x, y))

# Show Image
plt.figure(figsize=(8,8))
plt.imshow(image, cmap='gray')

# Draw bounding boxes
for (x, y) in detections:
    rect = plt.Rectangle(
        (x, y),
        window_size,
        window_size,
        edgecolor='red',
        facecolor='none',
        linewidth=2
    )
    plt.gca().add_patch(rect)

plt.title("R-CNN Concept Demonstration (Uploaded Image)")
plt.axis("off")
plt.show()
