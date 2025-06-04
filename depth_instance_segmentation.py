import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor

# Load depth estimation model and processor
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

# Load semantic segmentation model and processor
segmentation_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
segmentation_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

# Load the image
image_path = 'Your image path'  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Step 1: Depth Estimation
depth_inputs = depth_processor(images=image, return_tensors="pt")
with torch.no_grad():
    depth_outputs = depth_model(**depth_inputs)
    depth_map = depth_outputs.predicted_depth.squeeze().cpu().numpy()

# Normalize the depth map for better visualization
depth_min, depth_max = depth_map.min(), depth_map.max()
depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)

# Step 2: Semantic Segmentation
segmentation_inputs = segmentation_processor(images=image, return_tensors="pt")
with torch.no_grad():
    segmentation_outputs = segmentation_model(**segmentation_inputs)
    segmentation_map = torch.argmax(segmentation_outputs.logits.squeeze(), dim=0).cpu().numpy()

# Resize the depth map to match the segmentation map's dimensions
depth_map_resized = cv2.resize(depth_map, (segmentation_map.shape[1], segmentation_map.shape[0]))

# Cityscapes class labels
cityscapes_labels = {
    0: 'Road', 1: 'Sidewalk', 2: 'Building', 3: 'Wall', 4: 'Fence',
    5: 'Pole', 6: 'Traffic Light', 7: 'Traffic Sign', 8: 'Vegetation',
    9: 'Terrain', 10: 'Sky', 11: 'Person', 12: 'Rider', 13: 'Car',
    14: 'Truck', 15: 'Bus', 16: 'Train', 17: 'Motorcycle', 18: 'Bicycle'
}

# Define the Cityscapes class color map (standard for Cityscapes dataset)
cityscapes_colors = [
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (70, 70, 70),    # Building
    (102, 102, 156), # Wall
    (190, 153, 153), # Fence
    (153, 153, 153), # Pole
    (250, 170, 30),  # Traffic Light
    (220, 220, 0),   # Traffic Sign
    (107, 142, 35),  # Vegetation
    (152, 251, 152), # Terrain
    (70, 130, 180),  # Sky
    (220, 20, 60),   # Person
    (255, 0, 0),     # Rider
    (0, 0, 142),     # Car
    (0, 0, 70),      # Truck
    (0, 60, 100),    # Bus
    (0, 80, 100),    # Train
    (0, 0, 230),     # Motorcycle
    (119, 11, 32),   # Bicycle
]

# Map segmentation labels to RGB colors
segmentation_rgb = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
for class_id, color in enumerate(cityscapes_colors):
    segmentation_rgb[segmentation_map == class_id] = color

# Step 3: Collect depths for each object type (calculating the centroid and bounding box)
object_depths = []

# Create a copy of the segmentation map to draw contours
segmentation_rgb_with_boundaries = segmentation_rgb.copy()

# Create a color palette with unique colors for each object class
color_palette = {
    0: (0, 255, 0),    # Road (Green)
    1: (255, 0, 0),    # Sidewalk (Blue)
    2: (0, 0, 255),    # Building (Red)
    3: (255, 255, 0),  # Wall (Cyan)
    4: (255, 0, 255),  # Fence (Magenta)
    5: (0, 255, 255),  # Pole (Yellow)
    6: (128, 128, 128), # Traffic Light (Gray)
    7: (0, 128, 128),  # Traffic Sign (Dark Cyan)
    8: (128, 0, 128),  # Vegetation (Purple)
    9: (255, 165, 0),  # Terrain (Orange)
    10: (0, 255, 255), # Sky (Aqua)
    11: (255, 20, 147), # Person (Deep Pink)
    12: (0, 255, 127), # Rider (Spring Green)
    13: (255, 69, 0),  # Car (Red Orange)
    14: (255, 105, 180), # Truck (Hot Pink)
    15: (70, 130, 180),  # Bus (Steel Blue)
    16: (255, 99, 71),  # Train (Tomato)
    17: (255, 255, 224), # Motorcycle (Light Yellow)
    18: (0, 100, 0)     # Bicycle (Dark Green)
}

# Draw contours for each object and collect depths at centroids
for class_id, class_name in cityscapes_labels.items():
    mask = segmentation_map == class_id
    if np.any(mask):  # Check if the object is present in the image
        # Get coordinates of the object
        y_coords, x_coords = np.where(mask)

        # Calculate mean depth for the object
        mean_depth = depth_map_resized[mask].mean()

        # Calculate the centroid (center of mass)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # Get depth value at the centroid
        centroid_depth = depth_map_resized[center_y, center_x]

        object_depths.append({
            "Object": class_name,
            "Centroid": (center_x, center_y),
            "Depth at Centroid": centroid_depth
        })

        # Find the contour of the object
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw each contour with a unique color for each object
        for contour in contours:
            # Use the class-specific color for the contour
            color_bgr = color_palette.get(class_id, (255, 255, 255))  # Default to white if not found
            cv2.drawContours(segmentation_rgb_with_boundaries, [contour], -1, color_bgr, 3)  # Border line

    else:
        object_depths.append({
            "Object": class_name,
            "Centroid": "Not Present",
            "Depth at Centroid": "N/A"
        })

# Convert the results into a DataFrame
df_depths = pd.DataFrame(object_depths)

# Display the distance table
print("\nObject Depths at Centroids:")
print(df_depths.to_string(index=False))

# Visualization with centroids and boundaries marked
fig, ax = plt.subplots(1, 3, figsize=(20, 7))

# Original Image
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

# Depth Estimation
ax[1].imshow(depth_map_normalized, cmap='plasma')
ax[1].set_title("Depth Estimation")
ax[1].axis('off')

# Segmentation Map with Boundaries and Centroids
ax[2].imshow(segmentation_rgb_with_boundaries)
ax[2].set_title("Segmentation Map with Boundaries and Centroids")
ax[2].axis('off')

# Marking centroids
for obj in object_depths:
    if obj["Centroid"] != "Not Present":
        center_x, center_y = obj["Centroid"]
        ax[2].plot(center_x, center_y, 'r+', markersize=10)
        ax[2].text(center_x, center_y, obj["Object"], color='white', fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

# Create a color legend
handles = []
labels = []
for class_id, color in color_palette.items():
    label = f"{cityscapes_labels[class_id]}"
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color)/255., markersize=10))
    labels.append(label)

ax[2].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title="Classes")

plt.tight_layout()
plt.show()
