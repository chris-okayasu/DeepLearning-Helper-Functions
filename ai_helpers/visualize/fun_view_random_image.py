import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import numpy as np

def fun_view_random_image(target_dir, target_class, num_images=1, show_histogram=True):
    """
    Displays one or more random images from a given class, along with their color histograms.

    Parameters:
    - target_dir (str): Path to the main dataset directory.
    - target_class (str): Name of the class (subfolder).
    - num_images (int): Number of random images to display (default: 1).
    - show_histogram (bool): Whether to show histograms of pixel intensity (default: True).

    Returns:
    - Last image loaded as a NumPy array.
    """
    target_folder = os.path.join(target_dir, target_class)

    # Ensure the directory exists and contains images
    if not os.path.exists(target_folder) or not os.listdir(target_folder):
        print(f"Error: No images found in {target_folder}")
        return None

    # Select random images
    random_images = random.sample(os.listdir(target_folder), min(num_images, len(os.listdir(target_folder))))

    # Set up figure to display multiple images
    rows = 2 if show_histogram else 1
    fig, axes = plt.subplots(rows, num_images, figsize=(num_images * 3, rows * 3))

    if num_images == 1:  # If only one image, adjust axes to be iterable
        axes = np.array(axes).reshape(rows, -1)

    for i, img_name in enumerate(random_images):
        img_path = os.path.join(target_folder, img_name)
        img = mpimg.imread(img_path)

        # Display the image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{target_class} - {img_name}")
        axes[0, i].axis("off")

        # Display color histogram if enabled
        if show_histogram:
            if len(img.shape) == 3:  # RGB Image
                colors = ("red", "green", "blue")
                for j, color in enumerate(colors):
                    hist_data = img[:, :, j].ravel()
                    axes[1, i].hist(hist_data, bins=50, color=color, alpha=0.6, label=color)
                axes[1, i].legend()
            else:  # Grayscale Image
                hist_data = img.ravel()
                axes[1, i].hist(hist_data, bins=50, color="black", alpha=0.6)

            axes[1, i].set_title("Histogram")

        # Print image details
        print(f"\nImage: {img_name}")
        print(f"Shape: {img.shape}")
        print(f"Data Type: {img.dtype}")
        print(f"Min Pixel Value: {img.min()}, Max Pixel Value: {img.max()}")

    plt.tight_layout()
    plt.show()

    return img  # Return the last image loaded