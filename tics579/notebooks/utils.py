import matplotlib.pyplot as plt


def plot_number(idx, data):
    plt.imshow(data[idx]["X"].numpy().reshape(28, 28), cmap="gray")
    plt.title(f"Etiqueta: {data[idx]['y']}")
    plt.show()


def plot_training_curves(train_loss, validation_loss, n_epochs, title=""):
    plt.plot(
        range(1, n_epochs + 1),
        train_loss,
        label="Train Loss",
    )
    plt.plot(
        range(1, n_epochs + 1),
        validation_loss,
        label="Validation Loss",
    )
    plt.title(title)
    plt.legend()
    plt.show()
