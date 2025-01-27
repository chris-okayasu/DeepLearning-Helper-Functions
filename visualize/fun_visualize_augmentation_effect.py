import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def fun_visualize_augmentation_effect(train_dir, target_size=(224, 224)):
    """
    Visualizes the effect of augmentation by showing original and augmented images.

    Parameters:
    - train_dir (str): Path to the training dataset directory.
    - target_size (tuple): Target size for images.
    """
    # Create ImageDataGenerator with augmentation
    augment_datagen = ImageDataGenerator(
        rescale=1./255,           # Normalize pixel values to [0, 1]
        rotation_range=30,        # Randomly rotate images up to 30 degrees
        width_shift_range=0.2,    # Randomly shift images horizontally
        height_shift_range=0.2,   # Randomly shift images vertically
        shear_range=0.2,          # Apply random shear transformations
        zoom_range=0.2,           # Randomly zoom into images
        horizontal_flip=True,     # Randomly flip images horizontally
        fill_mode='nearest'       # Fill missing pixels after transformation
    )

    # Load a sample image from the training set
    class_name = random.choice(os.listdir(train_dir))  # Random class from the train folder
    class_folder = os.path.join(train_dir, class_name)
    img_name = random.choice(os.listdir(class_folder))  # Random image from that class
    img_path = os.path.join(class_folder, img_name)

    # Read the image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, target_size)

    # Convert the image from BGR to RGB for correct color display
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Show the original image
    plt.figure(figsize=(6, 6))
    plt.subplot(3, 3, 1)
    plt.imshow(img_resized_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # Prepare image for augmentation (add batch dimension)
    img_array = np.expand_dims(img_resized_rgb, axis=0)
    augmented_images = augment_datagen.flow(img_array, batch_size=1)

    # List of augmentation types to display as titles
    augmentations = [
        "Rotation", "Width Shift", "Height Shift", "Shear",
        "Zoom In", "Zoom Out", "Flip Horizontal", "Flip Vertical"
    ]

    # Plot augmented images and show the corresponding augmentation type
    for i in range(2, 10):
        augmented_img = next(augmented_images)[0]  # Get next augmented image using next()

        # Set the appropriate title for each augmentation type
        plt.subplot(3, 3, i)
        plt.imshow(augmented_img)

        # Display the augmentation type based on the index
        plt.title(f"{augmentations[i-2]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
