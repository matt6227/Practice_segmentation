# first attempt: will identify people but not basketball

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

# Load the pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Define the transformation to apply to the input frames
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to segment and highlight specific object classes
def segment_and_highlight_objects(model, image, target_classes):
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Create individual masks for each target class
    colored_masks = []
    for target_class in target_classes:
        object_mask = (output_predictions == target_class).astype(np.uint8) * 255
        colored_mask = cv2.applyColorMap(object_mask, cv2.COLORMAP_JET)
        colored_masks.append(colored_mask)

    # Combine all masks into a single image
    combined_mask = np.zeros_like(colored_masks[0])
    for mask in colored_masks:
        combined_mask = cv2.add(combined_mask, mask)

    return combined_mask

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Define the target class indices for the objects you want to highlight
target_classes = [0, 16, 41, 29]  # Example class indices (person, dog, basketball, frisbee)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to a smaller size for faster processing
    frame_resized = cv2.resize(frame, (640, 480))  # Adjust dimensions as needed

    # Segment and highlight the specific object classes
    highlighted_objects = segment_and_highlight_objects(model, frame_resized, target_classes)

    # Blend the original frame with the highlighted objects mask
    combined_frame = cv2.addWeighted(frame_resized, 0.6, highlighted_objects, 0.4, 0)

    cv2.imshow('Highlighted Objects', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
