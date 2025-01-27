# Another cool function a created to determinate and visualize a good learning rate...
import numpy as np
import matplotlib.pyplot as plt
def fun_plot_learning_rate(history, metric="loss", log_scale=True, ylim=None, save=False, save_path="lr_vs_loss.png", scheduler_type=None, base_lr=1e-4, max_lr=None, step_size=None, figsize=(14, 7)):
    """
    Visualiza la relación entre la tasa de aprendizaje y una métrica (por defecto, la pérdida),
    y ajusta automáticamente la longitud de las tasas de aprendizaje.

    Args:
        history (tf.keras.callbacks.History): Objeto de historial de entrenamiento.
        metric (str, optional): Métrica a graficar contra el learning rate. Por defecto es "loss".
        log_scale (bool, optional): Si True, usa escala logarítmica para el eje X. Por defecto es True.
        ylim (tuple, optional): Límites para el eje Y (métrica).
        save (bool, optional): Si True, guarda el gráfico como una imagen. Por defecto es False.
        save_path (str, optional): Ruta donde se guardará el gráfico si save=True. Por defecto es "lr_vs_loss.png".
        scheduler_type (str, optional): Tipo de scheduler a usar. Puede ser 'exponential_decay' o 'cyclic_lr'.
        base_lr (float, optional): Learning rate base para calcular la escala. Por defecto es 1e-4.
        max_lr (float, optional): El learning rate máximo para CyclicLR.
        step_size (int, optional): El paso para CyclicLR.
        figsize (tuple, optional): Tamaño del gráfico (ancho, alto). Por defecto es (14, 7).
    """
    # Verificar que la métrica existe en el historial
    if metric not in history.history:
        raise ValueError(f"La métrica '{metric}' no está en el historial. Métricas disponibles: {list(history.history.keys())}")

    # Extraer los valores de la métrica
    metric_values = history.history[metric]
    num_points = len(metric_values)

    # Calcular automáticamente las tasas de aprendizaje
    lrs = base_lr * (10 ** (tf.range(num_points) / 20))

    # Determinar el índice del mínimo valor de la métrica (punto ideal)
    min_loss_idx = np.argmin(metric_values)
    ideal_lr = lrs[min_loss_idx].numpy()  # Convertir a valor escalar
    ideal_loss = metric_values[min_loss_idx]

    # Configurar el gráfico
    plt.figure(figsize=figsize)  # Cambiar el tamaño del gráfico
    if log_scale:
        plt.semilogx(lrs, metric_values, label=f"Learning Rate vs. {metric.capitalize()}")
    else:
        plt.plot(lrs, metric_values, label=f"Learning Rate vs. {metric.capitalize()}")

    # Marcar el punto ideal con un marcador
    plt.scatter(ideal_lr, ideal_loss, color='red', marker='o', s=100, label=f"Punto Ideal (LR={ideal_lr:.5e}, Loss={ideal_loss:.5f})")

    # Etiquetas y título
    plt.xlabel("Learning Rate")
    plt.ylabel(metric.capitalize())
    plt.title(f"Learning Rate vs. {metric.capitalize()}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Configurar límites del eje Y si se especifican
    if ylim:
        plt.ylim(ylim)

    # Mostrar o guardar el gráfico
    if save:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico guardado en: {save_path}")
    plt.show()

    # Mostrar el valor del learning rate en formato Adam(lr=valor)
    print(f"\nEl learning rate ideal es: Adam(learning_rate={ideal_lr:.5e})")

    # Si se quiere usar un scheduler
    if scheduler_type:
        if scheduler_type == 'exponential_decay':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=ideal_lr,
                decay_steps=1000,
                decay_rate=0.9,
                staircase=True
            )
            print("\nUsando ExponentialDecay como scheduler.")
            return lr_schedule

        elif scheduler_type == 'cyclic_lr':
            if max_lr is None or step_size is None:
                raise ValueError("Para CyclicLR, max_lr y step_size son necesarios.")
            lr_schedule = CyclicLearningRate(ideal_lr, max_lr, step_size)
            print("\nUsando Cyclic Learning Rate como scheduler.")
            return lr_schedule

    # Devolver el learning rate ideal si no se especificó un scheduler
    return ideal_lr
