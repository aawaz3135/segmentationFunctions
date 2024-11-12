# segmentationFunctions

This repository contains Python code for processing 2D and 3D images, including functionalities for label growth and manipulation. The primary focus is on morphological operations such as growing labeled regions in volumetric data. These functions are designed to handle medical imaging data or any labeled datasets where precise region processing is required.

Features:
Grow Labels Function: Expand labeled regions in 2D or 3D with customizable parameters for growth direction and size.
Image Handling: Read and process .mhd files with metadata preservation.
Flexible Configurations: Adaptable for both slice-wise and volumetric operations.


Dependencies:
numpy
SimpleITK
scipy
matplotlib

Usage:
Integrate these functions into your image processing pipelines for tasks requiring region growth or label manipulations.