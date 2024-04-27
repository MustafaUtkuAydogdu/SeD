import os
import cv2
import numpy as np
import glob
import re
from tqdm import tqdm

def remove_scaling_factor(image_path):
    # Extract the base filename from the image path
    base_name = os.path.basename(image_path)
    
    # Define a regular expression pattern to match scaling factors (x2, x3, x4, x8)
    scaling_factor_pattern = re.compile(r'x[2348]')
    
    # Replace any matching scaling factors with an empty string
    clean_name = scaling_factor_pattern.sub('', base_name)
    
    return clean_name


def prepare_dataset(input_image_path, image_name, save_folder, crop_size, step_size, file_extension, tqdm_desc=None):
    # Read the input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    image_height, image_width = input_image.shape[0:2]

    # Generate coordinates for cropping
    #crop_coordinates_height = np.arange(0, image_height - crop_size + 1, step_size)
    # Calculate the starting positions for cropping along the height of the image
    start_positions_height = 0  # Starting pxosition for cropping
    end_positions_height = image_height - crop_size + 1  # Ending position for cropping

    # Generate a range of starting positions with a step size of step_size
    # This ensures that the cropping covers the entire height of the image
    # without going beyond its boundaries
    crop_coordinates_height = np.arange(start_positions_height, end_positions_height, step_size)

    if not image_height - (crop_coordinates_height[-1] + crop_size) > 0:
        print("Warning")
    else:
        crop_coordinates_height = np.append(crop_coordinates_height, image_height - crop_size)

    # Calculate the starting positions for cropping along the width of the image
    start_positions_width = 0  # Starting position for cropping
    end_positions_width = image_width - crop_size + 1  # Ending position for cropping

    # Generate a range of starting positions with a step size of step_size
    # This ensures that the cropping covers the entire width of the image
    # without going beyond its boundaries
    crop_coordinates_width = np.arange(start_positions_width, end_positions_width, step_size)
    if not image_width - (crop_coordinates_width[-1] + crop_size) > 0:
        print("Warning")
    else:
        crop_coordinates_width = np.append(crop_coordinates_width, image_width - crop_size)

    # Initialize index for cropped image naming
    cnt = 0

    # Extract the file extension

    # Loop through all possible cropping positions
    for start_height in tqdm(crop_coordinates_height, desc=tqdm_desc):
        for start_width in crop_coordinates_width:
            cnt += 1
            # Extract the cropped image
            cropped_image = input_image[start_height:start_height + crop_size, start_width:start_width + crop_size, ...]
            cropped_image = np.ascontiguousarray(cropped_image)

            # Construct output file path
            output_file_path = os.path.join(save_folder, f'{image_name}_crop{cnt:03d}{file_extension}')

            # Save the cropped image
            cv2.imwrite(output_file_path, cropped_image)


#first handle for high resolution images
#TODO: check for different crop and step size
crop_size_hr = 480
step_size_hr = 240

# Directory containing the input images
input_folder = "data/hr"

#save_folder = "dataset_cropped"
save_folder = "data/dataset_cropped/hr"

# Construct a file pattern to match PNG files in the input folder
file_pattern = os.path.join(input_folder, '*.png')

# Find all PNG files in the input folder
matching_files = glob.glob(file_pattern)

# Sort the list of file paths alphabetically
sorted_files = sorted(matching_files)

# Store the sorted list of file paths in img_list variable
img_list = sorted_files

# Display the list of sorted image paths
print("Sorted image paths:")
for img_path in img_list:
    print(img_path)
    img_name, file_extension = os.path.splitext(os.path.basename(img_path))
    image_name = remove_scaling_factor(img_name)
    prepare_dataset(img_path, image_name, save_folder, crop_size_hr, step_size_hr, file_extension, tqdm_desc="Processing HR images")


#handle for low resolution images
#TODO: check for different crop and step size
crop_size_lr = 120
step_size_lr = 60

# Directory containing the input images
input_folder = "data/lr"

#save_folder = "dataset_cropped"
save_folder = "data/dataset_cropped/lr"

# Construct a file pattern to match PNG files in the input folder
file_pattern = os.path.join(input_folder, '*.png')

# Find all PNG files in the input folder
matching_files = glob.glob(file_pattern)

# Sort the list of file paths alphabetically
sorted_files = sorted(matching_files)

# Store the sorted list of file paths in img_list variable
img_list = sorted_files

# Display the list of sorted image paths
print("Sorted image paths:")
for img_path in img_list:
    print(img_path)
    img_name, file_extension = os.path.splitext(os.path.basename(img_path))
    image_name = remove_scaling_factor(img_name)
    prepare_dataset(img_path, image_name, save_folder, crop_size_lr, step_size_lr, file_extension, tqdm_desc="Processing LR images")