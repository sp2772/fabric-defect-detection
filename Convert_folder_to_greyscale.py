import os
from PIL import Image
import numpy as np

# Function to process and save thresholded images
def process_and_save_images(input_folder, output_folder, threshold=200):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    i=0
    # Process each image
    for image_file in image_files:
        # if i== 200:
        #     break
        # Construct full image paths
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        
        # Open the image
        image = Image.open(input_path)
        
        # Convert to grayscale
        grayscale_image = image.convert("L")
        
        # Apply thresholding
        grayscale_array = np.array(grayscale_image)
        calculated_threshold = np.mean(grayscale_array) +6 #sensitivity fixed to 6 for all
        #for no_defect, hole  the threshold was 150
        #for horizontal,vertical,line threshold was 200, for stain threshhold was 230
        
        
        newthreshold = min(calculated_threshold, threshold)
        print("new Threshold:",newthreshold)
        binary_image_array = (grayscale_array > newthreshold).astype(np.uint8) * 255
        binary_image = Image.fromarray(binary_image_array)
        
        # Save the binary image to the output folder
        binary_image.save(output_path)
        print(f"Processed and saved: {output_path}")
        i+=1

    print("number of processed:",i)
    
# Example usage
input_folder = "C:/MyEverything/PythonProjects/Recent_projects/cnn_analysis/MiniProjectCNN/Fabric Defect Dataset/Unprocessed_dataset/lines" # Replace with your input folder path
output_folder = "C:/MyEverything/PythonProjects/Recent_projects/cnn_analysis/MiniProjectCNN/Fabric Defect Dataset/lines_processed"  # Replace with your output folder path
process_and_save_images(input_folder, output_folder)
