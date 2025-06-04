**Depth Estimation and Semantic Segmentation Project**

**Project Overview**
This project demonstrates the use of deep learning models for:

Depth Estimation of a scene.
Semantic Segmentation to label objects in an image and compute their depths.

**Key Highlights**
Depth Estimation Model: Intel DPT Large
Semantic Segmentation Model: NVIDIA SegFormer
Input: An RGB image

**Outputs:**
Depth Map visualized using a plasma colormap.
Segmentation Map with boundaries and centroids for objects.
Tabular Report displaying object types and their depth values.

**Setup and Usage Instructions**

**1. Prerequisites**
Ensure you have the following installed:

Python (version 3.8 or higher).
pip for installing Python libraries.
Familiarity with the terminal or command line.

**2. Clone or Upload the Project**
Create a project directory on your local machine or Swecha.
Add the following files to your project directory:
The Python script: depth_segmentation.py

The test image file(s):sampleimage

**3. Directory Structure**
Organize your project directory as follows:


```
depth-estimation-segmentation/
│
├── depth_segmentation.py       # Main script
├── requirements.txt            # Dependencies
├── test_images/                # Folder for input images
│   ├── image1.jpg
│   ├── image2.png
│   └── ...                     # Add all your test images here
└── outputs/                    # Outputs (will be created after running the script)
```

**4. Install Required Dependencies**
Open a terminal in the project directory.
Run the following command to install the required Python libraries:

`pip install -r requirements.txt`

**5**. **Verify the Python Script**
Open the depth_segmentation.py file and verify:
Image Path: Ensure the correct file name and path in the script:


`image_path = 'Your image path'`

change this path with your image path

Ensure indentation and syntax are correct.

**6. Run the Project**
Run the Python script using this command:

`python depth_segmentation.py`

**Model Details**
**Depth Estimation:**
Model: Intel DPT Large
Description: Pre-trained on large-scale datasets for precise depth estimation.
Source: Intel Open Model Zoo.

**Semantic Segmentation:**
Model: NVIDIA SegFormer
Description: Lightweight transformer-based model designed for efficient segmentation.
Source: NVIDIA Research.

**Outputs**
**1. Depth Map**
Displays the depth of the scene with a color gradient (e.g., plasma colormap).

**2. Segmentation Map**
Assigns class labels to objects in the image (e.g., Road, Car, Building).
Marks boundaries and centroids for each detected object.

**3. Object Depth Table**
A table listing the following for each detected object:

Object Name: (e.g., Road, Car).
Centroid: Coordinates of the object’s center.
Depth at Centroid: Average depth value at the centroid.

Example Output Table:
| Object | Centroid  |Depth at Centroid |
|--------|-----------|------------------|
|Road    |(100, 200) |    	3.56        |
|--------|-----------|__________________|
|Building| (250, 450)|       8.74       |
|--------|-----------|__________________|



**Example Visual Outputs**
**Depth Map:**
Displays a gradient map showing the distance of objects from the camera.

**Segmentation Map:**
Highlights objects in the scene with labeled boundaries and centroid markers.

For output reference please see the sampleoutput file

**Key Commands Recap**
Install Dependencies:

`pip install -r requirements.txt`

Run the Script:

`python depth_segmentation.py`

Share the Project

**Distribute the following files:**


depth_segmentation.py

requirements.txt

About this Project:

This project leverages advanced deep learning techniques to perform **depth estimation** and **semantic segmentation** on images, with a strong focus on disaster response and scene analysis. Using state-of-the-art models—Intel DPT Large for depth estimation and NVIDIA SegFormer for semantic segmentation—the system processes an input RGB image to produce:

- A **depth map** that visualizes the distance of objects from the camera.
- A **segmentation map** that labels each pixel according to object class (e.g., road, building, car, as well as disaster-specific classes like collapsed buildings, fallen trees, and debris).
- A **tabular report** listing detected objects, their centroid coordinates, and their depth values.
- A **semantic relationship graph** that maps the spatial structure and proximity of objects within the scene.

**Key Applications:**
- **Disaster Assessment:** By comparing pre- and post-disaster images, the project can identify and map changes such as collapsed buildings and blocked roads, supporting rapid and automated damage assessment. This approach aligns with recent research that highlights the value of combining segmentation and classification for post-disaster management, enabling faster, more objective, and large-scale analysis compared to manual methods[1][2].
- **Emergency Response Prioritization:** The semantic graph and depth data help emergency teams quickly locate and prioritize aid to the most affected or inaccessible areas, improving the efficiency of rescue and relief operations.
- **Urban and Infrastructure Monitoring:** Beyond disaster scenarios, the system can be used for ongoing monitoring of urban environments, infrastructure, and environmental changes.

**Technical Highlights:**
- **Automated, pixel-level scene understanding** using deep neural networks.
- **Support for disaster-specific object classes** to enhance relevance in real-world crisis scenarios.
- **Visual and structural outputs** (maps, tables, graphs) that are actionable and interpretable for both technical and non-technical stakeholders.

In summary, this project provides a robust, automated framework for extracting actionable insights from images in disaster and urban contexts, helping to accelerate and improve the accuracy of critical decision-making processes.

This README.md file for guidance.
Ensure users follow the directory structure and instructions provided above for seamless execution.
 
 project Done by 

 Name           : University Id 
 
 B. Ajith kumar :2200080242

 N. Sai krishna :2200080251

 K.Visanth Keerthan Sai:2200080236
