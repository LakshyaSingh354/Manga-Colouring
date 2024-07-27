import cv2
import os
import numpy as np

def is_black_and_white(image_path, threshold=0.9):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale image back to BGR
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Calculate the difference between the original image and the grayscale image
    difference = cv2.absdiff(image, gray_bgr)
    
    # Calculate the mean of the difference
    mean_diff = np.mean(difference)
    
    # If the mean difference is below a certain threshold, consider it black and white
    return mean_diff < threshold

def delete_black_and_white_images(directory, threshold=0.9):
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(directory, filename)
            if is_black_and_white(image_path, threshold):
                print(f"Deleting black and white image: {filename}")
                os.remove(image_path)

# Directory containing your images
image_directory = "images/"

# Delete black and white images from the directory
delete_black_and_white_images(image_directory)