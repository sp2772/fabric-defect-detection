import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np

# Normalize the image to range [0, 255]
def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def process_images(folder):
    # Get all image files from folder
    images = [f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Select 5 random images
    selected_images = random.sample(images, 10)
    
    plt.figure(figsize=(20, 20))
    
    for img_index, img_name in enumerate(selected_images):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Noise removal using Gaussian Blur
        noise_removed = cv2.GaussianBlur(img, (51, 51), 0)
        noise_removed = normalize_image(noise_removed)  # Normalize after Gaussian blur
        
        # Apply Sobel Operator for edge detection
        sobelx = cv2.Sobel(noise_removed, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(noise_removed, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = normalize_image(sobel_combined)  # Normalize after Sobel
        
        #sobel_combined_inverted = cv2.bitwise_not(sobel_combined)
        # Apply second Gaussian Blur
        noise_removed2 = cv2.GaussianBlur(sobel_combined, (51, 51), 0)
        noise_removed2 = normalize_image(noise_removed2)  # Normalize after second blur
        
        # Sobel on denoised result
        sobelx1 = cv2.Sobel(noise_removed2, cv2.CV_64F, 1, 0, ksize=7)
        sobely1 = cv2.Sobel(noise_removed2, cv2.CV_64F, 0, 1, ksize=7)
        sobel_combined2 = cv2.magnitude(sobelx1, sobely1)
        sobel_combined2 = normalize_image(sobel_combined2)  # Normalize after second Sobel
        
        # Display original image
        plt.subplot(10, 5, img_index * 5 + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original {img_index+1}")
        plt.axis('off')
        
        # Display noise removed image
        plt.subplot(10, 5, img_index * 5 + 2)
        plt.imshow(noise_removed, cmap='gray')
        plt.title(f"Denoised {img_index+1}")
        plt.axis('off')
        
        # Display Sobel edge detection result
        plt.subplot(10, 5, img_index * 5 + 3)
        plt.imshow(sobel_combined, cmap='gray')
        plt.title(f"Sobel {img_index+1}")
        plt.axis('off')
        
        # Display noise removed image
        plt.subplot(10, 5, img_index * 5 + 4)
        plt.imshow(noise_removed2, cmap='gray')
        plt.title(f"DenoisedEX {img_index+1}")
        plt.axis('off')
        
        # Display Sobel on denoised image
        plt.subplot(10, 5, img_index * 5 + 5)
        plt.imshow(sobel_combined2, cmap='gray')
        plt.title(f"SobelEx {img_index+1}")
        plt.axis('off')
    
    plt.show()

# Set folder X where images are stored
#folder_X = r"C:\MyEverything\PythonProjects\Recent_projects\cnn_analysis\MiniProjectCNN\Dataset\PROCESSED_SET\PROCESSED_TRAINANDTESTSETS\NewTrainset_processed\Defect"
home_dir = os.path.expanduser("~")  # Gets the home directory
folder_X= os.path.join(home_dir, "Dataset/RAW_TRAIN_SET/Defect")
print(folder_X)
plt.switch_backend('TkAgg') 
process_images(folder_X)
