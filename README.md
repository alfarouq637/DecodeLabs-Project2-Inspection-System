# DecodeLabs Robotics & Automation Internship - PROJECT 2 ⚙️

[<img width="1672" height="941" alt="ChatGPT Image May 30, 2026, 12_46_28 PM" src="https://github.com/user-attachments/assets/b30f2bb7-d152-4a19-ae9d-df41371ca2c5" />
](https://youtu.be/akEHXhRGcZo)


**Developer:** Alfarouq Ibrahim | Robotics & Automation Intern  
**Project:** Multi-Modal Edge-AI Inspection System for Industrial Gears

### Project Demo
[![Watch the system in action](<img width="1672" height="941" alt="ChatGPT Image May 30, 2026, 12_46_28 PM" src="https://github.com/user-attachments/assets/b30f2bb7-d152-4a19-ae9d-df41371ca2c5" />
)](https://youtu.be/akEHXhRGcZo)



## 📌 Project Overview
This project is an advanced Edge-AI research prototype developed during the **DecodeLabs Robotics & Automation Internship**. The objective is to automate the inspection of mechanical gears on an assembly line, detecting structural defects (like broken teeth) with high industrial reliability.

Instead of relying solely on classical Computer Vision (which is sensitive to lighting and reflections), this project implements a **Multi-Modal Explainable AI (XAI) Fusion Architecture**.

## ✨ Key Features & Upgrades

### 1. Pure Math Custom CNN (No Frameworks)
Built a Convolutional Neural Network entirely from scratch using **NumPy**. It handles forward propagation, backpropagation, and cross-entropy loss without relying on heavy frameworks like PyTorch or TensorFlow, making it highly optimized for Edge-devices.

### 2. Geometric Sonification (FFT Signal AI)
Inspired by **[Insert Name Here]**, the system includes a data-independent physical engine. It extracts the gear's outer contour, unrolls it into a 1D signal, and uses **Fast Fourier Transform (FFT)** to detect harmonic distortions caused by broken teeth.

### 3. Explainable Late Fusion (XAI Dashboard)
The system doesn't act as a black box. It features a custom green-terminal dashboard that displays three independent evaluations:
* **Classical PLC Gate:** Deterministic OpenCV logic (Gaussian Blur, Thresholding, Convexity Defects) with dynamic bounding boxes targeting only the gear teeth.
* **Visual AI:** The NumPy CNN prediction.
* **Signal AI:** The FFT mathematical prediction.
* **Final Fused Decision:** A confidence-weighted late fusion combining all modalities for maximum fault tolerance.

### 4. Procedural 3D Dataset Generation
Due to the lack of high-fidelity industrial datasets, a custom Python script was written for **Blender (Cycles Engine)**. It uses Involute Curve mathematics to procedurally generate over 2,000 photorealistic 1080p images of intact and defective Carbon Steel gears.

## 📂 Repository Structure
* `main.py`: The core fusion engine and dashboard UI.
* `scratch_cnn.py`: The custom NumPy-based CNN architecture.
* `geometric_sonification.py`: The FFT signal extraction and analysis model.
* `fusion_coordinator.py`: The late-fusion logic combining CNN and FFT probabilities.
* `METHODOLOGY_AND_ARCHITECTURE.md`: Formal academic documentation of the architecture.
* `gear_data/`: The generated dataset (Not fully uploaded due to size limits. Use `--prepare-data` to initialize).

## 🙏 Credits

DecodeLabs: For the internship opportunity and the classical CV pipeline baseline.

Kai Zhou, Jiong Tang: For the concept and inspiration behind the FFT/Spectral Signal analysis module.
(https://data.mendeley.com/datasets/87y47nvsf4/1)

## 🚀 How to Run

1. Clone the repository and install requirements:
```powershell
pip install -r requirements.txt
2.Run the Inference Dashboard (Ensure you have downloaded the weights gear_cnn_hd.npz and some test images in gear_data/unlabeled):
python main.py --load-model gear_cnn_hd.npz --epochs 0 --image-size 96 --infer-dir gear_data/unlabeled



