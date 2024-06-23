
# Reality Capture and 3D Modelling

Welcome to the repository for the Feasibility Study on Precise 3D Models of Cars in Intelligent Transportation Systems (ITS). This research project aims to bridge the knowledge gap caused by the limited Fields of View (FOVs) of surveillance cameras in ITS, which provide incomplete information about vehicles. By leveraging photogrammetry, a technique that combines images from multiple perspectives, we propose a method to create accurate and complete 3D models of observed vehicles. These models will enhance vehicle recognition, tracking, and provide valuable insights into traffic patterns and behavior. Join us on this journey to contribute to advancements in road safety, emergency response systems, and traffic management.

## Project Overview

This project focuses on creating precise 3D models of vehicles using multiple video streams and advanced photogrammetry techniques. The main objectives include:

- Enhancing vehicle recognition and tracking in ITS.
- Overcoming challenges posed by moving cameras and varying lighting conditions.
- Optimizing computational efficiency for real-time processing.
- Providing valuable insights into traffic patterns and vehicle behavior.

## Methodology

### Key Components
- **YOLOv5**: Used for object detection to identify and locate vehicles in video frames.
- **MIDAS**: Employed for depth estimation to provide 3D information from 2D images.
- **OpenCV**: Utilized for various image processing tasks, including 3D reconstruction.

### Steps Involved
1. **Image Capture**: Collecting sufficient photos from multiple angles.
2. **Pre-processing**: Applying image correction, feature detection (using FAST algorithm), and depth estimation.
3. **3D Reconstruction**: Using OpenCV to create 3D models from processed images.

## Installation

Ensure you have Python 3.9.13 installed. Follow these steps to set up the project:

1. Clone the repositories:
   ```bash
   git clone https://github.com/ozgekarasu/ENS491-RealityCapture.git
   git clone https://github.com/ultralytics/yolov5

2. Navigate to the YOLOv5 directory:
   ```bash
   cd yolov5

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

4. Install additional dependencies:
   ```bash
   pip install torch
   pip install opencv-contrib-python
   pip install numpy
   pip install scipy
   pip install open3d
   
**Note:** If you encounter issues with installing PyTorch, you can download it directly from the [official PyTorch website](https://pytorch.org/).

## References

- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- [Midas Repository](https://github.com/isl-org/MiDaS)
- [Computer Vision by Nico Nielsen](https://github.com/niconielsen32/ComputerVision)

## Academic References
@article{Ranftl2020,
	author    = {René Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}

@article{Ranftl2021,
	author    = {René Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}

## Authors

- İpek Nur Dolu: [@ipekdolu](https://github.com/ipekdolu)
- Özge Karasu: [@ozgekarasu](https://github.com/ozgekarasu)
- Serkan Kütük: [@serkanktk](https://github.com/serkanktk)

## Appendix

![Ekran Görüntüsü (3933)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/8e2985b3-235b-47e0-b2ce-a7198a9666a6)



![Ekran Görüntüsü (3934)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/64b1fc17-dd9e-473a-bacf-d1fa5d253ec5)


![Ekran Görüntüsü (3935)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/424510ee-9407-4743-b1e2-d40a9a3fb2e8)


![Ekran Görüntüsü (3936)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/6ec392b3-497b-4bc6-af76-e6b95f113de0)


![Ekran Görüntüsü (3937)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/81df3ada-461f-4bce-8e8d-221c13bcb156)



![Ekran Görüntüsü (3938)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/b1c91faf-639d-42a2-be5b-94253dc4c516)



![Ekran Görüntüsü (3939)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/8042d77e-14f5-4c18-899b-535be7edab07)



![Ekran Görüntüsü (3940)](https://github.com/ozgekarasu/ENS491-RealityCapture/assets/128151657/ff6d1e26-ec4e-492e-908a-37243daff2b9)






