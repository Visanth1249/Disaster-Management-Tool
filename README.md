# Depth Estimation & Semantic Segmentation Project

## 🚀 Project Overview

This project harnesses the power of state-of-the-art deep learning models to **analyze, understand, and quantify scenes from RGB images**. By combining **depth estimation** and **semantic segmentation**, it enables you to not only label objects in a scene but also determine their spatial depth—unlocking a wide range of applications from robotics to disaster response.

---

## 🌟 Key Features

- **Depth Estimation**: Leveraging the Intel DPT Large model, the project generates high-resolution depth maps, visualized with a vivid plasma colormap for intuitive understanding.
- **Semantic Segmentation**: Utilizing NVIDIA’s SegFormer, it accurately labels and segments objects within an image, drawing boundaries and pinpointing centroids.
- **Comprehensive Reporting**: Outputs a detailed table summarizing each detected object’s type, centroid coordinates, and average depth value.
- **Visual Outputs**: Generates easy-to-interpret maps and overlays for both depth and segmentation results.

---

## 🛠️ Setup & Usage Instructions

### 1. Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: For installing dependencies
- **Basic command line skills**

### 2. Clone or Upload the Project

Create a directory on your local machine or on [Swecha](https://swecha.org/) and add the following files:

- `depth_segmentation.py` (main script)
- `requirements.txt` (dependencies)
- Place your test images in the `test_images/` folder

### 3. Directory Structure

```
depth-estimation-segmentation/
│
├── depth_segmentation.py       # Main script
├── requirements.txt            # Dependencies
├── test_images/                # Input images
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── outputs/                    # Outputs (created after running the script)
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Script

- **Set the Image Path**:  
  Open `depth_segmentation.py` and update the following line to point to your image:
  ```python
  image_path = 'test_images/your_image.jpg'
  ```
- **Check Syntax**:  
  Ensure indentation and syntax are correct for your environment.

### 6. Run the Project

```bash
python depth_segmentation.py
```

---

## 🧠 Model Details

- **Depth Estimation**
  - **Model**: Intel DPT Large
  - **Source**: [Intel Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
  - **Description**: Trained on large-scale datasets for highly accurate depth predictions.

- **Semantic Segmentation**
  - **Model**: NVIDIA SegFormer
  - **Source**: [NVIDIA Research](https://github.com/NVlabs/SegFormer)
  - **Description**: Lightweight transformer-based model for efficient and precise segmentation.

---

## 📊 Outputs

### 1. Depth Map
- Visualizes the scene’s depth using a plasma color gradient, making it easy to interpret distances at a glance.

### 2. Segmentation Map
- Assigns class labels (e.g., Road, Car, Building) to each object.
- Draws clear boundaries and marks centroids for each detected object.

### 3. Object Depth Table

| Object    | Centroid   | Depth at Centroid |
|-----------|------------|-------------------|
| Road      | (100, 200) | 3.56              |
| Building  | (250, 450) | 8.74              |

*(See the `sampleoutput` file for reference visualizations.)*

---

## 🌍 Real-World Applications

This project is not just a technical demo—it’s a practical tool for:

- **Disaster Response**: Compare pre- and post-disaster images to generate semantic graphs, identify damaged areas, and prioritize emergency aid based on object changes and depth analysis.
- **Autonomous Vehicles & Robotics**: Enhance scene understanding for navigation and obstacle avoidance.
- **Smart City Monitoring**: Track changes in infrastructure and urban environments over time.

---

## 📝 Key Commands Recap

- **Install Dependencies**:  
  `pip install -r requirements.txt`
- **Run the Script**:  
  `python depth_segmentation.py`

---

## 🤝 Share & Collaborate

When sharing this project, include:

- `depth_segmentation.py`
- `requirements.txt`
- A few sample images in `test_images/`
- This `README.md` for guidance

**Tip:** Always follow the directory structure and instructions for a seamless experience!

---

## 💡 Contributing

Pull requests, issues, and suggestions are welcome! Help us make this project even more impactful.

---

## 📬 Contact

For questions, collaborations, or support, open an issue or reach out via GitHub discussions.

---

**Empower your images with AI-driven scene understanding—depth, semantics, and actionable insights, all in one place!**

---
