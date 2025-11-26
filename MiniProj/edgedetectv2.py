import cv2
import os
import random
import matplotlib.pyplot as plt

def process_images(folder):
    # Get all image files from folder
    plt.switch_backend('TkAgg') 
    images = [f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Select 5 random images
    selected_images = random.sample(images, 5)
    
    # Different Sobel kernel sizes
    kernel_sizes = [3, 5, 7]
    
    plt.figure(figsize=(20, 20))
    
    for img_index, img_name in enumerate(selected_images):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Noise removal using Gaussian Blur
        noise_removed = cv2.GaussianBlur(img, (101, 101), 0)
        
        # Display original image
        plt.subplot(5, 4, img_index * 4 + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original {img_index+1}")
        plt.axis('off')
        
        # Display noise removed image
        plt.subplot(5, 4, img_index * 4 + 2)
        plt.imshow(noise_removed, cmap='gray')
        plt.title(f"Denoised {img_index+1}")
        plt.axis('off')
        
        # Apply and display Sobel edge detection for different kernel sizes
        for i, ksize in enumerate(kernel_sizes):
            # sobelx = cv2.Sobel(noise_removed, cv2.CV_64F, 1, 0, ksize=ksize)
            # sobely = cv2.Sobel(noise_removed, cv2.CV_64F, 0, 1, ksize=ksize)
            # sobel_combined = cv2.magnitude(sobelx, sobely)
            
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel_combined = cv2.magnitude(sobelx, sobely)
            
            plt.subplot(5, 5, img_index * 5 + 3 + i)
            plt.imshow(sobel_combined, cmap='gray')
            plt.title(f"Sobel k={ksize}")
            plt.axis('off')
    
    plt.show()

# Set folder X where images are stored
home_dir = os.path.expanduser("~")  # Gets the home directory
folder_X= os.path.join(home_dir, "Dataset/RAW_TRAIN_SET/Defect")
print(folder_X)
process_images(folder_X)
