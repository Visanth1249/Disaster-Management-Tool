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

# Cityscapes class labels
cityscapes_labels = {
    0: 'Road', 1: 'Sidewalk', 2: 'Building', 3: 'Wall', 4: 'Fence',
    5: 'Pole', 6: 'Traffic Light', 7: 'Traffic Sign', 8: 'Vegetation',
    9: 'Terrain', 10: 'Sky', 11: 'Person', 12: 'Rider', 13: 'Car',
    14: 'Truck', 15: 'Bus', 16: 'Train', 17: 'Motorcycle', 18: 'Bicycle'
}

# Define color palette for segmentation visualization
color_palette = {
    0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 128, 128), 7: (0, 128, 128),
    8: (128, 0, 128), 9: (255, 165, 0), 10: (0, 255, 255), 11: (255, 20, 147),
    12: (0, 255, 127), 13: (255, 69, 0), 14: (255, 105, 180), 15: (70, 130, 180),
    16: (255, 99, 71), 17: (255, 255, 224), 18: (0, 100, 0)
}

# Function to run full analysis on an image
def analyze_scene(image_path):
    image = Image.open(image_path).convert("RGB")

    # Depth Estimation
    depth_inputs = depth_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        depth_outputs = depth_model(**depth_inputs)
        depth_map = depth_outputs.predicted_depth.squeeze().cpu().numpy()

    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)

    # Semantic Segmentation
    segmentation_inputs = segmentation_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        segmentation_outputs = segmentation_model(**segmentation_inputs)
        segmentation_map = torch.argmax(segmentation_outputs.logits.squeeze(), dim=0).cpu().numpy()

    # Resize depth to match segmentation
    depth_map_resized = cv2.resize(depth_map, (segmentation_map.shape[1], segmentation_map.shape[0]))

    # Object-level data extraction
    object_data = {}
    for class_id, class_name in cityscapes_labels.items():
        mask = segmentation_map == class_id
        if np.any(mask):
            y_coords, x_coords = np.where(mask)
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            centroid_depth = depth_map_resized[center_y, center_x]
            object_data[class_name] = {
                "centroid": (center_x, center_y),
                "depth": float(centroid_depth),
                "area": len(x_coords)
            }
    return object_data, depth_map_normalized, segmentation_map, image

# Function to compare two scenes and compute a priority score
def compare_scenes(pre_data, post_data):
    pre_obj, _, _, _ = pre_data
    post_obj, _, _, _ = post_data

    priority_scores = []
    all_classes = set(pre_obj.keys()).union(post_obj.keys())

    for cls in all_classes:
        pre = pre_obj.get(cls)
        post = post_obj.get(cls)

        if pre and not post:
            score = 1.0  # Object missing completely
        elif pre and post:
            depth_diff = abs(pre['depth'] - post['depth'])
            area_diff_ratio = abs(pre['area'] - post['area']) / (pre['area'] + 1e-5)
            score = 0.5 * depth_diff + 0.5 * area_diff_ratio
        else:
            continue

        priority_scores.append((cls, score))

    # Sort by score descending
    priority_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n🚨 Priority List for Rescue & Relief 🚨")
    print("{:<20} {:>10}".format("Object Class", "Priority Score"))
    print("-" * 32)
    for cls, score in priority_scores:
        print(f"{cls:<20} {score:>10.3f}")

# Function to visualize a single scene (optional for visual analysis)
def visualize_scene(image, depth_map_normalized, segmentation_map, title="Scene Visualization"):
    segmentation_rgb = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_palette.items():
        segmentation_rgb[segmentation_map == class_id] = color

    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(depth_map_normalized, cmap='plasma')
    ax[1].set_title("Depth Map")
    ax[1].axis('off')

    ax[2].imshow(segmentation_rgb)
    ax[2].set_title("Segmentation Map")
    ax[2].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# -------------------- MAIN EXECUTION --------------------

# Provide your actual image paths here
pre_disaster_image = 'test_images/pre_disaster.jpg'
post_disaster_image = 'test_images/post_disaster.jpg'

print("\n🔎 Analyzing Pre-Disaster Scene...")
pre_data = analyze_scene(pre_disaster_image)
visualize_scene(*pre_data[3:], title="Pre-Disaster Scene")

print("\n🔎 Analyzing Post-Disaster Scene...")
post_data = analyze_scene(post_disaster_image)
visualize_scene(*post_data[3:], title="Post-Disaster Scene")

# Run comparison
compare_scenes(pre_data, post_data)
