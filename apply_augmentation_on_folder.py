import os
from PIL import Image, ImageEnhance
import numpy as np
import random
import shutil

# Function to apply augmentations
def augment_image(image, augment_num):
    augmented_images = []
    
    for i in range(augment_num):
        # Make a copy of the image
        aug_image = image.copy()
        
        # Apply random augmentations
        if random.choice([True, False,True]):
            aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal Flip
        if random.choice([True, False,True]):
            aug_image = aug_image.transpose(Image.FLIP_TOP_BOTTOM)  # Vertical Flip
        if random.choice([True, False,True]):
            angle = random.randint(-20, 20)  # Random rotation(for horizontal)
            aug_image = aug_image.rotate(angle)
        if random.choice([True, False,True]):
            zoom_factor = random.uniform(0.8, 1.2)  # Random zoom
            w, h = aug_image.size
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            aug_image = aug_image.resize((new_w, new_h), Image.LANCZOS)
            if zoom_factor > 1:
                # Crop back to original size if zoomed in
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                aug_image = aug_image.crop((left, top, left + w, top + h))
            else:
                # Pad if zoomed out
                aug_image = aug_image.crop((0, 0, w, h))
        if random.choice([True, False,True]):
            enhancer = ImageEnhance.Brightness(aug_image)
            aug_image = enhancer.enhance(random.uniform(0.5, 1.5))  # Random brightness
        if random.choice([True, False,True]):
            enhancer = ImageEnhance.Contrast(aug_image)
            aug_image = enhancer.enhance(random.uniform(0.5, 1.5))  # Random contrast
        

        # Add to the augmented images list
        augmented_images.append(aug_image)

    return augmented_images

# Function to fill images up to 1000
def augment_images_to_1000(input_folder, output_folder, target_count=900):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Copy original images to output folder
    for image_file in image_files:
        shutil.copy(os.path.join(input_folder, image_file), os.path.join(output_folder, image_file))
    
    # Check how many images are needed
    original_count = len(image_files)
    augment_num = (target_count - original_count) // original_count + 1  # Approx number of augmentations per image
    
    print(f"Original images: {original_count}, augmentations per image: {augment_num}")
    
    # Process and augment each image
    for image_file in image_files:
        # Open the original image
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)
        
        # Generate augmented images
        augmented_images = augment_image(image, augment_num)
        
        # Save augmented images
        for idx, aug_image in enumerate(augmented_images):
            # Save with new name
            base_name, ext = os.path.splitext(image_file)
            aug_name = f"{base_name}_augment{idx+1}{ext}"
            aug_image.save(os.path.join(output_folder, aug_name))

    # Check final count
    final_count = len(os.listdir(output_folder))
    print(f"Final number of images in '{output_folder}': {final_count}")

# Example usage
input_folder = "C:/MyEverything/PythonProjects/Recent_projects/cnn_analysis/MiniProjectCNN/Fabric Defect Dataset/Processed_dataset/lines_processed"  # Replace with your input folder
output_folder = "C:/MyEverything/PythonProjects/Recent_projects/cnn_analysis/MiniProjectCNN/Fabric Defect Dataset/Processed_dataset/lines_augmented_processed"  # Replace with your output folder
augment_images_to_1000(input_folder, output_folder)
