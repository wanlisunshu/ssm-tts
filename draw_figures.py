import matplotlib.pyplot as plt
import numpy as np
import os


def draw_loss_figure(loss_list, iter_list, key_word, save_path):
    fig = plt.figure()
    ax = fig.add_subplot()

    train_label_list = ["loss", "first loss", "second loss"]
    marker_list = ["kx-", "r.-", "b.-"]
    for i, loss in enumerate(loss_list):
        ax.plot(iter_list, loss, marker_list[i], label=train_label_list[i])


    ax.set_title("Training Loss")
    ax.set_xlabel(key_word)
    ax.set_ylabel("loss")
    ax.legend()

    figure_name = ("loss_" + key_word)
    figure_path = os.path.join(save_path, figure_name)
    fig.savefig(figure_path)
