Here's the **edited version** of your project documentation with the requested update. The new section elaborates on how **pre- and post-disaster images** are compared through a **graph-based evaluation** to determine areas of maximum destruction and prioritize emergency response:

---

# **Depth Estimation and Semantic Segmentation Project**

## **Project Overview**

This project demonstrates the use of deep learning models for:

* **Depth Estimation** of a scene
* **Semantic Segmentation** to label objects in an image and compute their depths

## **Key Highlights**

* **Depth Estimation Model**: Intel DPT Large
* **Semantic Segmentation Model**: NVIDIA SegFormer
* **Input**: An RGB image

## **Outputs:**

* Depth Map visualized using a plasma colormap
* Segmentation Map with boundaries and centroids for objects
* Tabular Report displaying object types and their depth values

---

## **Setup and Usage Instructions**

### **1. Prerequisites**

Ensure you have the following installed:

* Python (version 3.8 or higher)
* pip for installing Python libraries
* Familiarity with the terminal or command line

### **2. Clone or Upload the Project**

Create a project directory on your local machine or Swecha.
Add the following files to your project directory:

* `depth_segmentation.py` (Main Python script)
* Test image file(s) in the `test_images/` folder

### **3. Directory Structure**

```
depth-estimation-segmentation/
│
├── depth_segmentation.py       # Main script
├── requirements.txt            # Dependencies
├── test_images/                # Folder for input images
│   ├── image1.jpg
│   ├── image2.png
│   └── ...                     # Add your test images here
└── outputs/                    # Output folder (auto-created)
```

### **4. Install Required Dependencies**

Run this command in the terminal inside the project directory:

```bash
pip install -r requirements.txt
```

### **5. Verify the Python Script**

Open `depth_segmentation.py` and update:

```python
image_path = 'Your image path'  # <-- Replace with your image path
```

Ensure indentation and syntax are correct.

### **6. Run the Project**

```bash
python depth_segmentation.py
```

---

## **Model Details**

### **Depth Estimation**

* **Model**: Intel DPT Large
* **Description**: Pre-trained model for accurate depth prediction
* **Source**: Intel Open Model Zoo

### **Semantic Segmentation**

* **Model**: NVIDIA SegFormer
* **Description**: Lightweight transformer-based model for efficient pixel-wise classification
* **Source**: NVIDIA Research

---

## **Outputs**

### **1. Depth Map**

A gradient color-coded image (plasma colormap) showing distance of objects from the camera.

### **2. Segmentation Map**

Pixel-wise class labels (e.g., Road, Car, Building) with boundaries and centroids marked.

### **3. Object Depth Table**

A table listing:

* **Object Name** (e.g., Road, Car)
* **Centroid Coordinates**
* **Depth at Centroid**

**Example Table:**

| Object   | Centroid   | Depth at Centroid |
| -------- | ---------- | ----------------- |
| Road     | (100, 200) | 3.56              |
| Building | (250, 450) | 8.74              |

---

## **Example Visual Outputs**

* **Depth Map**: Gradient map visualizing object distances.
* **Segmentation Map**: Labeled objects with boundaries and centroids.

---

## **Disaster Comparison and Prioritization Logic**

After generating depth maps and segmentation masks, the system allows for **comparison between a pre-disaster image and a post-disaster image**.

### **Key Features of Disaster Comparison:**

* **Semantic Graph Comparison**:

  * The scene is converted into a graph with objects as nodes.
  * Edges represent spatial relationships and proximities.
  * Pre- and post-disaster graphs are compared to detect structural changes.

* **Evaluation Metrics**:

  * Number of demolished vs. intact structures
  * Degree of object disappearance (e.g., collapsed buildings, blocked roads)
  * Shifts in object centroids or depth values

### **Graph-Based Evaluation**

Each region is evaluated by comparing:

* **Before Disaster Graph (G₁)**
* **After Disaster Graph (G₂)**

A **difference score** is computed based on:

* Number of destroyed structures
* Lost connectivity (e.g., roads cut off)
* Depth changes indicating structural collapse

### **Rescue Prioritization**

Areas with the **highest damage scores** are flagged for **emergency response**:

* **Ambulance Dispatch**
* **Firefighter Units**
* **Rescue and Relief Teams**

This allows **smart, data-driven prioritization** in critical disaster scenarios.

---

## **Key Applications**

* **Disaster Assessment**
* **Emergency Response Prioritization**
* **Urban & Infrastructure Monitoring**

---

## **Technical Highlights**

* State-of-the-art deep learning with transformer-based segmentation
* Pixel-level scene understanding
* Disaster-aware object class handling
* Graph-based evaluation for real-time prioritization

---

## **How to Share the Project**

Distribute the following files:

* `depth_segmentation.py`
* `requirements.txt`
* `README.md` (this guide)

---

## **Credits**

**Project Done By:**

| Name                    | University ID |
| ----------------------- | ------------- |
| B. Ajith Kumar          | 2200080242    |
| N. Sai Krishna          | 2200080251    |
| K. Visanth Keerthan Sai | 2200080236    |

---
