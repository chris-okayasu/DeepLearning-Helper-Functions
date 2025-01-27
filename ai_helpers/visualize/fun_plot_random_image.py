import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def plot_random_image(
    model,
    images,
    true_labels,
    classes,
    input_shape=(28, 28, 1),  # Expected shape of the input
    batch_size=1,  # Number of random images to display
    title_size=16,
    cmap="binary",
    save_images=False,  # Whether to save the images
    save_dir="output_images",  # Directory to save images
    show_probabilities=True  # Whether to show class probabilities for each image
):
    """
    Selects random images or a batch of images from the dataset, predicts their labels using the model,
    and visualizes them with the true labels and predicted labels. Optionally saves the images and displays
    class probabilities.

    Args:
        model (tf.keras.Model): Trained model for making predictions.
        images (np.ndarray): Array of images to pick from.
        true_labels (np.ndarray): Array of true labels corresponding to the images.
        classes (list): List of class names corresponding to label indices.
        input_shape (tuple, optional): Expected shape of the input images. Default is (28, 28, 1).
        batch_size (int, optional): Number of random images to display. Default is 1.
        title_size (int, optional): Font size of the title. Default is 16.
        cmap (str, optional): Colormap for the images. Default is "binary".
        save_images (bool, optional): If True, saves the visualized images. Default is False.
        save_dir (str, optional): Directory to save images if save_images=True. Default is "output_images".
        show_probabilities (bool, optional): If True, shows the probability distribution for each image. Default is True.
    """
    # Validate inputs
    if len(images) != len(true_labels):
        raise ValueError("The number of images and true labels must be the same.")
    if images[0].shape[:2] != input_shape[:2]:
        raise ValueError(
            f"Images must have shape {input_shape[:2]}, but got {images[0].shape[:2]}"
        )
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    # Ensure the save directory exists if save_images is True
    if save_images and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Select random indices
    random_indices = random.sample(range(len(images)), batch_size)

    # Set up the figure dimensions (one row per batch item, two columns: image + probabilities)
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, batch_size * 6))
    if batch_size == 1:
        axes = [axes]  # Make axes iterable for single batch case

    for idx, i in enumerate(random_indices):
        # Retrieve the target image and true label
        target_image = images[i]
        true_label = int(true_labels[i])

        # Prepare the image for prediction (add batch dimension if necessary)
        target_image_input = target_image[np.newaxis, ...]

        # Predict probabilities
        pred_probs = model.predict(target_image_input, verbose=0)
        pred_label_idx = np.argmax(pred_probs)
        pred_label = classes[pred_label_idx]
        pred_confidence = pred_probs[0][pred_label_idx] * 100

        # Determine title color
        color = "green" if pred_label_idx == true_label else "red"

        # Plot the image on the left
        axes[idx][0].imshow(target_image, cmap=cmap)
        axes[idx][0].axis(False)
        axes[idx][0].set_title(
            f"Pred: {pred_label} ({pred_confidence:.2f}%)\nTrue: {classes[true_label]}",
            fontsize=title_size,
            color=color,
        )

        # Plot the probabilities on the right
        if show_probabilities:
            axes[idx][1].bar(classes, pred_probs[0], color="blue")
            axes[idx][1].set_ylim(0, 1)  # Probabilities are between 0 and 1
            axes[idx][1].set_title("Class Probabilities", fontsize=title_size)
            axes[idx][1].set_ylabel("Probability")
            axes[idx][1].tick_params(axis='x', rotation=45)
        else:
            axes[idx][1].axis(False)  # If probabilities are not shown, hide the subplot

        # Save each image if save_images is True
        if save_images:
            save_path = os.path.join(save_dir, f"image_{i}_pred_{pred_label}.png")
            plt.savefig(save_path, dpi=300)
            print(f"Saved image {i} to {save_path}")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
