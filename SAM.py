# does not really work both images look the same 




# from segment_anything import sam_model_registry
# sam = sam_model_registry["vit_h"](checkpoint="/Users/matthewsoto/Stanford/ARMSLab/practice_open_cv/sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
# predictor.set_image('/Users/matthewsoto/Pictures/German-Shepherd-dog-Alsatian.webp')
# masks, _, _ = predictor.predict('dog')


# ATTEMPT 2

# from segment_anything import sam_model_registry, SamPredictor
# import cv2  # OpenCV for image processing

# # Load the SAM model
# checkpoint_path = "/Users/matthewsoto/Stanford/ARMSLab/practice_open_cv/sam_vit_h_4b8939.pth"
# sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)

# # Initialize the predictor
# predictor = SamPredictor(sam)

# # Read an image using OpenCV
# image_path = "/Users/matthewsoto/Pictures/German-Shepherd-dog-Alsatian.webp"
# image = cv2.imread(image_path)

# # Check if the image was loaded successfully
# if image is None:
#     raise ValueError(f"Image at path {image_path} could not be loaded. Check the path and file format.")

# # Convert the image to RGB (if needed)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Set the image in the predictor
# predictor.set_image(image)

# # Example prediction (adjust according to your specific needs)
# # result = predictor.predict(...)

# # Print a success message
# print("Prediction completed successfully")

########################################################
# Attempt 3 with image


# from segment_anything import sam_model_registry, SamPredictor
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the SAM model
# checkpoint_path = "/Users/matthewsoto/Stanford/ARMSLab/practice_open_cv/sam_vit_h_4b8939.pth"
# try:
#     sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
#     print("SAM model loaded successfully.")
# except Exception as e:
#     print(f"Error loading SAM model: {e}")

# # Initialize the predictor
# try:
#     predictor = SamPredictor(sam)
#     print("Predictor initialized successfully.")
# except Exception as e:
#     print(f"Error initializing predictor: {e}")

# # Read an image using OpenCV
# image_path = "/Users/matthewsoto/Pictures/German-Shepherd-dog-Alsatian.webp"
# image = cv2.imread(image_path)

# if image is None:
#     raise ValueError(f"Image at path {image_path} could not be loaded. Check the path and file format.")
# else:
#     print(f"Image loaded successfully from {image_path}.")

# # Convert the image to RGB
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Set the image in the predictor
# try:
#     predictor.set_image(image)
#     print("Image set in predictor successfully.")
# except Exception as e:
#     print(f"Error setting image in predictor: {e}")

# # Perform the prediction
# try:
#     masks, _, _ = predictor.predict()
#     print("Prediction completed successfully.")
# except Exception as e:
#     print(f"Error during prediction: {e}")

# # Check if masks were generated
# if masks is not None and len(masks) > 0:
#     print(f"Number of masks predicted: {len(masks)}")
#     mask = masks[0]  # Use the first mask for visualization
    
#     # Ensure the mask values are in the range [0, 1]
#     if mask.max() > 1:
#         mask = mask / 255.0
    
#     print(f"Mask shape: {mask.shape}, Mask max value: {mask.max()}, Mask min value: {mask.min()}")
    
#     # Create an overlay with transparency
#     alpha = 0.5  # Transparency factor
#     overlay = image.copy()
#     for c in range(3):
#         overlay[:, :, c] = np.where(mask > 0, overlay[:, :, c] * (1 - alpha) + alpha * 255, overlay[:, :, c])
    
#     # Display the original image and the overlay
#     plt.figure(figsize=(10, 10))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Original Image")
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(overlay)
#     plt.title("Image with Mask Overlay")

#     plt.show()
# else:
#     print("No masks were predicted.")

# ################################################################
# ATTEMPT 4

import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load the SAM model
model_type = "vit_h"
checkpoint_path = "/Users/matthewsoto/Stanford/ARMSLab/practice_open_cv/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
device = "cpu"
sam.to(device=device)

# Initialize the mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Initialize video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    if masks and len(masks) > 0:
        mask = masks[0]['segmentation']
        
        # Ensure the mask values are in the range [0, 1]
        if mask.max() > 1:
            mask = mask / 255.0

        # Create an overlay with transparency
        alpha = 0.5  # Transparency factor
        overlay = image.copy()
        for c in range(3):
            overlay[:, :, c] = np.where(mask > 0, overlay[:, :, c] * (1 - alpha) + alpha * 255, overlay[:, :, c])

        # Convert overlay back to BGR for OpenCV
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Display the frame with overlay
        cv2.imshow('Live Camera Stream with Mask Overlay', overlay_bgr)
    else:
        # If no masks, just display the original frame
        cv2.imshow('Live Camera Stream', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
