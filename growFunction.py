
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from pathlib import Path
import os
import time
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from scipy.ndimage import zoom

def grow_labels(label_image, fixed_labels, working_labels, grow_mode, grow_amount):
    """
    Grows specified labels in a 3D label image by a given amount of pixels.

    Parameters:
    - label_image (np.ndarray): 3D array of labeled regions with integer labels.
    - fixed_labels (list or np.ndarray): Labels that should remain fixed and unaffected.
    - working_labels (list or np.ndarray): Labels that will be grown.
    - grow_mode (str): Growth mode, either '2D' for slice-wise growth or '3D' for volumetric growth.
    - grow_amount (int): Number of dilation iterations to grow the working labels.

    Returns:
    - np.ndarray: New label image with grown labels.
    """
    # Copy the original label image
    new_label_image = np.copy(label_image)

    # Create binary masks for fixed and working labels
    fixed_mask = np.isin(label_image, fixed_labels)
    working_mask = np.isin(label_image, working_labels)

    # Define the structuring element for dilation
    if grow_mode == '2D':
        struct_element = np.ones((3, 3, 1), dtype=bool)  # Growth in the xy-plane
    elif grow_mode == '3D':
        struct_element = np.ones((3, 3, 3), dtype=bool)  # Growth in all directions
    else:
        raise ValueError("grow_mode must be '2D' or '3D'")

    # Perform label growth for each label in working_labels
    for label in working_labels:
        label_mask = (label_image == label)  # Binary mask for the current label
        grown_mask = binary_dilation(label_mask, structure=struct_element, iterations=grow_amount)

        # Prevent growth into fixed labels
        grown_mask &= ~fixed_mask

        # Update the new label image with the grown mask
        new_label_image[grown_mask] = label

    return new_label_image



def read_image(full_path_label,visualize=False):
    """
    Reads  labeled images from .mhd files and converts them to numpy arrays.

    Parameters:
    - visualize (bool): If True, visualizes a slice of the grayscale and labeled images.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Grayscale image array and labeled image array.
    """
   
   

    # Read grayscale and label images using SimpleITK
    reader = sitk.ImageFileReader()
    #reader.SetFileName(full_path_gray)
    #grayscale = sitk.GetArrayFromImage(reader.Execute())  # Convert to numpy array

    reader.SetFileName(full_path_label)
    label = sitk.GetArrayFromImage(reader.Execute())  # Convert to numpy array

    # Visualize the middle slice if requested
    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        #ax[0].imshow(grayscale[grayscale.shape[0] // 2], cmap="gray")
        #ax[0].set_title("Raw Image Slice")
        ax[0].imshow(label[label.shape[0] // 2], cmap="nipy_spectral")
        ax[0].set_title("Labeled Image Slice")
        pores_fractures_array = (label == 0).astype(np.uint8)
        ax[1].imshow(pores_fractures_array[pores_fractures_array.shape[0] // 2], cmap="PuOr")
        ax[1].set_title("Isolated Segment")
        plt.show()
        #asds

    return  label





# Main code
if __name__ == "__main__":

    # Define paths
    root_dir = Path(__file__).parent
    dataset = 'D5623/subsetImage/newSubset'
    #gray_scale_image = 'D5602-SEM-2_1_slices_1_cropped_multiply_ml_slices_ver5_grayscale.mhd'
    label_image_name = 'D5623-xrm-2.7um-BKRM_ml_seg-e_200_slices.mhd'
    pathName = Path(os.path.normpath(root_dir)[0:-28]) / dataset
    folderName=''
    full_path_label = pathName /  folderName /label_image_name #Supply the path for label image

    #This function reads the label image in numpy array
    label_image = read_image(full_path_label,visualize=False)  
   

    # Define fixed and working labels
    fixed_labels = [0]  # Background
    working_labels = [1]  # Labels to grow

    # Grow labels in 3D with specified iterations
    grow_mode = '3D'
    grow_amount = 1

    start_time = time.time()  # Start the timer

    # Call the grow_labels function
    new_label_image = grow_labels(label_image, fixed_labels, working_labels, grow_mode, grow_amount)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"grow_labels executed in {elapsed_time:.4f} seconds.")
    
    
    #Visualize result here
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(label_image[0], cmap="PuOr") #This plot just one slice based on the input slice
    ax[0].set_title("Before Growth")
    ax[1].imshow(new_label_image[0], cmap="PuOr")
    ax[1].set_title("After Growth")
    plt.show()
    