#identifies still image of people but can change the label to detect other items


import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2

# # Load a pre-trained Mask R-CNN model
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.eval()  # Set the model to evaluation mode

# # Load an image
# img_path = '/Users/matthewsoto/Stanford/ARMSLab/practice_open_cv/dog_image.jpeg'
# img = Image.open(img_path).convert("RGB")

# # # matt: adjusting the image
# # target_width = 800
# # target_height = 600
# # img_resized = cv2.resize(img, (target_width, target_height))


# # Preprocess the image
# transform = T.Compose([T.ToTensor()])
# img = transform(img)

# # Add a batch dimension
# img = img.unsqueeze(0)

# # Perform inference
# with torch.no_grad():
#     prediction = model(img)

# # Visualize the results
# def plot_image(image, masks, boxes):
#     fig, ax = plt.subplots(1, figsize=(12,9))
#     ax.imshow(image)
#     for i in range(len(masks)):
#         mask = masks[i].cpu().numpy()
#         box = boxes[i].cpu().numpy()
#         ax.imshow(mask, alpha=0.5)
#         rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
#         ax.add_patch(rect)
#     plt.show()

# # Convert image back to numpy
# img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

# # Plot the image with masks and bounding boxes
# plot_image(img_np, prediction[0]['masks'], prediction[0]['boxes'])



# # second attempt

# import cv2
# # import matplotlib.pyplot as plt

# # Path to the image file
# image_path = '/Users/matthewsoto/Stanford/ARMSLab/practice_open_cv/matt_steph_tayna.jpeg'

# # Load image using OpenCV
# img = cv2.imread(image_path)

# # Check if image is loaded successfully
# if img is None:
#     print(f"Error: Unable to load image at {image_path}")
# else:
#     # Resize image to a smaller size for display
#     img_resized = cv2.resize(img, (800, 600))  # Adjust dimensions as needed

#     # Convert BGR to RGB (OpenCV loads images as BGR by default)
#     img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

#     # Display the image using Matplotlib
#     plt.imshow(img_rgb)
#     plt.axis('off')  # Hide axis labels
#     plt.show()

###

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transformation to apply to the input image
transform = T.Compose([
    T.ToTensor(),
])

# Path to the image file
# image_path = '/Users/matthewsoto/Stanford/ARMSLab/practice_open_cv/dog_image.jpeg'
image_path = '/Users/matthewsoto/Pictures/German-Shepherd-dog-Alsatian.webp'


# Load image using OpenCV
img = cv2.imread(image_path)

# Check if image is loaded successfully
if img is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Resize image to a smaller size for processing
    target_width = 800
    target_height = 600
    img_resized = cv2.resize(img, (target_width, target_height))

    # Convert BGR to RGB (OpenCV loads images as BGR by default)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Transform and normalize image
    input_tensor = transform(img_rgb)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Get predictions
    with torch.no_grad():
        prediction = model(input_batch)

    # Plot the image and overlay masks or bounding boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)

    # Iterate over detected instances
    for i in range(len(prediction[0]['labels'])):
        label = prediction[0]['labels'][i].item()
        if label == 1:  # Assuming label 1 corresponds to person
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            img_rgb[mask == 1] = [0, 255, 0]  # Highlight in green

    plt.axis('off')
    plt.imshow(img_rgb)
    plt.show()
