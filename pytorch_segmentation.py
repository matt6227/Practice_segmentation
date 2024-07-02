import torch
import torchvision
import cv2
import numpy as np

# Load a DeepLabV3 model pretrained on COCO dataset
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set model to evaluation mode

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Define a function to apply color map to the segmentation mask
def decode_segmentation(mask, colormap):
    """Decode segmentation mask into an RGB image."""
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for l in range(0, len(colormap)):
        idx = mask == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Color map for COCO dataset (21 classes, you can customize it)
colormap = [
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128)  # TV/monitor
]

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame for the model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    input_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]  # Get the segmentation mask
        output_predictions = output.argmax(0).cpu().numpy()  # Get the predicted class for each pixel
    
    # Decode the segmentation mask
    decoded_mask = decode_segmentation(output_predictions, colormap)
    
    # Resize the decoded mask to match the original frame size
    decoded_mask_resized = cv2.resize(decoded_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Blend the original frame with the segmentation mask
    blended_frame = cv2.addWeighted(frame, 0.5, decoded_mask_resized, 0.5, 0)
    
    # Display the result
    cv2.imshow('Live Segmentation', blended_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
