import matplotlib.pyplot as plt
import numpy as np
import csv


# plot one file
def plot_file(title: str, in_file: str, out_file: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    epoch = []
    train_loss = []
    train_sd = []
    accuracy = []
    first = True
    with open(in_file, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            if first:
                first = False
            else:
                epoch.append(float(row[0]))
                train_loss.append(float(row[1]))
                train_sd.append(float(row[2]))
                accuracy.append(float(row[3]))
    epoch = np.array(epoch)
    train_loss = np.array(train_loss)
    train_sd = np.array(train_sd)
    # Plot training loss
    plt.ylim(0, 1)
    ax.plot(epoch, train_loss, c='g', label='Training Loss')
    ax.fill_between(epoch, train_loss - train_sd, train_loss + train_sd,
                    color='g', alpha=0.2)
    # plot accuracy
    ax.plot(epoch, accuracy, c='r', label='Validation Accuracy')
    plt.title(f"Training Loss vs Validation Accuracy: {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


for lr in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
           0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    plot_file(f'lr = {lr}', f'../results_{lr}.csv', f'plot_{lr}.pdf')
    plot_file(f'lr = {lr}, simple model', f'../results_simple_{lr}.csv', f'plot_simple_{lr}.pdf')
    plot_file(f'lr = {lr}, complex model', f'../results_complex_{lr}.csv', f'plot_complex_{lr}.pdf')
