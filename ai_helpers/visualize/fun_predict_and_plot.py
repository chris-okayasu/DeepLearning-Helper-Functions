import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fun_predict_and_plot(model, image_path, class_names, top_n=3, target_size=(224, 224), save=False, save_path="prediction.png"):
    """
    Loads an image, preprocesses it, makes a prediction using the given model,
    and displays the image with the predicted class label, confidence score,
    and a bar chart of the top predictions.

    Args:
        model (tf.keras.Model): Trained model for prediction.
        image_path (str): Path to the image.
        class_names (list): List of class names corresponding to model output indices.
        top_n (int, optional): Number of top predictions to display in the bar chart. Default is 3.
        target_size (tuple, optional): Target size for image resizing. Default is (224, 224).
        save (bool, optional): Whether to save the plot as an image file. Default is False.
        save_path (str, optional): File path to save the plot. Default is "prediction.png".
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions

    # Make prediction
    predictions = model.predict(img_array)[0]
    top_indices = np.argsort(predictions)[-top_n:][::-1]  # Get top N predictions
    top_labels = [class_names[i] for i in top_indices]
    top_confidences = [predictions[i] for i in top_indices]
    colors = sns.color_palette("husl", top_n)  # Generate distinct colors

    # Create figure with image and bar chart
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Show the image
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title(f"Prediction: {top_labels[0]} ({top_confidences[0]:.2f})")

    # Show bar chart of top predictions
    ax[1].barh(top_labels[::-1], top_confidences[::-1], color=colors)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("Confidence")
    ax[1].set_title("Top Predictions")

    plt.tight_layout()

    # Save the plot if requested
    if save:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()
