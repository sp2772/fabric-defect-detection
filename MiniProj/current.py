import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Enable dynamic memory allocation
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3850)]  # Set memory limit in MB
            )
        print("GPU memory growth enabled and memory limit set.")
    except RuntimeError as e:
        print(e)
train_dir = "/mnt/c/256x256Train/train/MultiClass"

test_dir = "/mnt/c/256x256Train/test/MultiClass"

# Function to apply Sobel filters
def apply_sobel_filters(image):
    """
    Apply Sobel filters to detect horizontal and vertical edges
    
    Args:
        image: Input grayscale image
        
    Returns:
        sobel_x: Gradient in X direction (detects vertical edges)
        sobel_y: Gradient in Y direction (detects horizontal edges)
    """
    # Apply Sobel operator in x direction (detects vertical edges)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in y direction (detects horizontal edges)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize the gradient images
    sobel_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return sobel_x, sobel_y

# Function to create augmented data generator with Sobel features
def create_augmented_data_generator(train_dir, batch_size, img_size, color_mode='grayscale'):
    """
    Create a data generator with Sobel filter features
    
    Args:
        train_dir: Directory containing training images
        batch_size: Batch size
        img_size: Image dimensions (height, width)
        color_mode: 'grayscale' or 'rgb'
        
    Returns:
        A generator that yields batches of images with Sobel features
    """
    # Original data generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
    )
    
    # Create a generator for original images
    original_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode
    )
    
    while True:
        # Get the next batch of images and labels
        X_batch, y_batch = next(original_generator)
        
        # Create a new batch with additional channels for Sobel features
        if color_mode == 'grayscale':
            # For grayscale images
            X_new_batch = np.zeros((X_batch.shape[0], img_size[0], img_size[1], 3))
            
            for i in range(X_batch.shape[0]):
                # Get the original grayscale image
                gray_img = X_batch[i, :, :, 0]
                
                # Apply Sobel filters
                sobel_x, sobel_y = apply_sobel_filters(gray_img)
                
                # Create a 3-channel image: [original, sobel_x, sobel_y]
                X_new_batch[i, :, :, 0] = gray_img
                X_new_batch[i, :, :, 1] = sobel_x / 255.0  # Normalize to [0,1]
                X_new_batch[i, :, :, 2] = sobel_y / 255.0  # Normalize to [0,1]
        else:
            # For RGB images
            X_new_batch = np.zeros((X_batch.shape[0], img_size[0], img_size[1], 5))
            
            for i in range(X_batch.shape[0]):
                # Get the original RGB image
                rgb_img = X_batch[i]
                
                # Convert to grayscale for Sobel filtering
                gray_img = np.mean(rgb_img, axis=2)
                
                # Apply Sobel filters
                sobel_x, sobel_y = apply_sobel_filters(gray_img)
                
                # Create a 5-channel image: [R, G, B, sobel_x, sobel_y]
                X_new_batch[i, :, :, 0:3] = rgb_img
                X_new_batch[i, :, :, 3] = sobel_x / 255.0  # Normalize to [0,1]
                X_new_batch[i, :, :, 4] = sobel_y / 255.0  # Normalize to [0,1]
        
        yield X_new_batch, y_batch

# Function to create the CNN model
def create_model(input_shape, num_classes):
    """
    Create a CNN model that accepts images with Sobel features
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to classify line direction based on Sobel responses
def classify_line_direction(image):
    """
    Classify whether an image contains horizontal or vertical line defects
    based on Sobel filter responses
    
    Args:
        image: Input grayscale image
        
    Returns:
        direction: 'horizontal', 'vertical', or 'none'
        confidence: Confidence score for the classification
    """
    # Apply Sobel filters
    sobel_x, sobel_y = apply_sobel_filters(image)
    
    # Calculate the sum of gradient magnitudes in each direction
    sum_x = np.sum(sobel_x)  # Vertical edges (horizontal lines)
    sum_y = np.sum(sobel_y)  # Horizontal edges (vertical lines)
    
    # Calculate the ratio of horizontal to vertical edges
    total = sum_x + sum_y
    if total == 0:
        return 'none', 0.0
    
    ratio_x = sum_x / total  # Proportion of vertical edges
    ratio_y = sum_y / total  # Proportion of horizontal edges
    
    # Determine the dominant direction
    threshold = 0.6  # Threshold for considering a direction dominant
    
    if ratio_x > threshold:
        return 'horizontal', ratio_x  # More vertical edges indicate horizontal lines
    elif ratio_y > threshold:
        return 'vertical', ratio_y    # More horizontal edges indicate vertical lines
    else:
        return 'none', max(ratio_x, ratio_y)

# Main execution code
if __name__ == "__main__":
    # Image parameters
    IMG_SIZE = (256, 256)  # Image size
    BATCH_SIZE = 40  # Number of images in each batch
    
    # Create data generators with Sobel features
    train_generator = create_augmented_data_generator(
        train_dir, 
        BATCH_SIZE, 
        IMG_SIZE, 
        color_mode='grayscale'
    )
    
    test_generator = create_augmented_data_generator(
        test_dir, 
        BATCH_SIZE, 
        IMG_SIZE, 
        color_mode='grayscale'
    )
    
    # Get class information
    temp_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale'
    )
    
    print("Class indices:", temp_generator.class_indices)
    print("Number of Classes:", temp_generator.num_classes)
    print("Class distribution:", Counter(temp_generator.classes))
    
    # Create and compile the model
    NUM_CLASSES = len(temp_generator.class_indices)
    INPUT_CHANNELS = 3  # Original grayscale + 2 Sobel channels
    
    model = create_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], INPUT_CHANNELS),
        num_classes=NUM_CLASSES
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=14,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=6,
        min_lr=0.00001
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model_with_sobel.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(temp_generator.classes) // BATCH_SIZE,
        epochs=50,
        validation_data=test_generator,
        validation_steps=len(temp_generator.classes) // BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history_with_sobel.png')
    plt.show()