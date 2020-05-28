import matplotlib.pyplot as plt
import numpy as np
import csv
import os


# plot one file
def plot_file(title: str, in_file: str, out_file: str) -> None:
    if not os.path.isfile(in_file):
        print(f"Warning: could not find {in_file}")
        return
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
    plt.suptitle(f"Training Loss vs Validation Accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


for lr in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
           0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    lr2 = str(lr).replace('.', '_')
    plot_file(f'lr = {lr}', f'../results_{lr}.csv', f'plot_{lr2}.pdf')
    plot_file(f'lr = {lr}, simple model', f'../results_simple_{lr}.csv', f'plot_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, complex model', f'../results_complex_{lr}.csv', f'plot_complex_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.1', f'../results_wd0.1_{lr}.csv', f'plot_wd0_1_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.1, simple model', f'../results_wd0.1_simple_{lr}.csv', f'plot_wd0_1_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.1, complex model', f'../results_wd0.1_complex_{lr}.csv', f'plot_wd0_1_complex_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.01', f'../results_wd0.01_{lr}.csv', f'plot_wd0_01_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.01, simple model', f'../results_wd0.01_simple_{lr}.csv', f'plot_wd0_01_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.01, complex model', f'../results_wd0.01_complex_{lr}.csv', f'plot_wd0_01_complex_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.001', f'../results_wd0.001_{lr}.csv', f'plot_wd0_001_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.001, simple model', f'../results_wd0.001_simple_{lr}.csv', f'plot_wd0_001_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, wd=0.001, complex model', f'../results_wd0.001_complex_{lr}.csv', f'plot_wd0_001_complex_{lr2}.pdf')
    plot_file(f'lr = {lr}, simple scaling', f'../results_op1_{lr}.csv', f'plot_op1_{lr2}.pdf')
    plot_file(f'lr = {lr}, simple scaling, simple model', f'../results_op1_simple_{lr}.csv', f'plot_op1_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, simple scaling, complex model', f'../results_op1_complex_{lr}.csv', f'plot_op1_complex_{lr2}.pdf')
    plot_file(f'lr = {lr}, augmented', f'../results_augmented_{lr}.csv', f'plot_augmented_{lr2}.pdf')
    plot_file(f'lr = {lr}, augmented, simple model', f'../results_augmented_simple_{lr}.csv', f'plot_augmented_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, augmented, complex model', f'../results_augmented_complex_{lr}.csv', f'plot_augmented_complex_{lr2}.pdf')
    plot_file(f'lr = {lr}, 10% dropout', f'../results_dropout0.1_{lr}.csv', f'plot_dropout10_{lr2}.pdf')
    plot_file(f'lr = {lr}, 10% dropout, simple model', f'../results_dropout0.1_simple_{lr}.csv', f'plot_dropout10_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, 10% dropout, complex model', f'../results_dropout0.1_complex_{lr}.csv', f'plot_dropout10_complex_{lr2}.pdf')
    plot_file(f'lr = {lr}, 20% dropout', f'../results_dropout0.2_{lr}.csv', f'plot_dropout20_{lr2}.pdf')
    plot_file(f'lr = {lr}, 20% dropout, simple model', f'../results_dropout0.2_simple_{lr}.csv', f'plot_dropout20_simple_{lr2}.pdf')
    plot_file(f'lr = {lr}, 20% dropout, complex model', f'../results_dropout0.2_complex_{lr}.csv', f'plot_dropout20_complex_{lr2}.pdf')
