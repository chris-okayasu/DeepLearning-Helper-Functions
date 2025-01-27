# Create a confusion matrix as def for using in other projects
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def fun_plot_confusion_matrix(y_true, y_pred, classes=None, normalize=True, cmap="Blues", figsize=(10, 10), title="Confusion Matrix", text_size=15, save=False, save_path="confusion_matrix.png"):
    """
    Dibuja una matriz de confusión con opciones para personalizar el diseño.
    """
    # Alinear formatos de y_true y y_pred
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:  # Si y_pred es multietiqueta
        y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:  # Si y_true es multietiqueta
        y_true = np.argmax(y_true, axis=1)

    # Crear la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    n_classes = cm.shape[0]  # Número de clases

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.get_cmap(cmap))
    fig.colorbar(cax)

    # Etiquetas para los ejes
    if classes is None:
        classes = np.arange(n_classes)

    ax.set(title=title,
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=classes,
           yticklabels=classes)

    # Poner etiquetas del eje x en la parte inferior
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Ajustar tamaño del texto
    ax.title.set_size(20)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Umbral para el color del texto
    threshold = (cm.max() + cm.min()) / 2.

    # Mostrar texto en las celdas
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        value = f"{cm[i, j]}"  # Valores absolutos
        if normalize:
            value += f" ({cm_norm[i, j] * 100:.1f}%)"  # Agregar valores normalizados
        ax.text(j, i, value,
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)

    # Ajustar el diseño
    plt.tight_layout()

    # Guardar la figura si se especifica
    if save:
        plt.savefig(save_path, dpi=300)
        print(f"Matriz de confusión guardada en: {save_path}")

    plt.show()
