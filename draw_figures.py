import matplotlib.pyplot as plt
import numpy as np
import os


def draw_iter_loss_figure(loss_list, iter_list, key_word, save_path):
    fig = plt.figure()
    ax = fig.add_subplot()

    train_label_list = ["loss", "first loss", "second loss"]
    marker_list = ["kx-", "r.-", "b.-"]
    for i, loss in enumerate(loss_list):
        ax.plot(iter_list, loss, marker_list[i], label=train_label_list[i])


    ax.set_title("Training Loss")
    ax.set_xlabel(key_word)
    ax.set_ylabel("loss")
    ax.set_yscale("symlog")
    ax.legend()

    figure_name = ("loss_" + key_word)
    figure_path = os.path.join(save_path, figure_name)
    fig.savefig(figure_path)


def draw_loss_figures(train_epoch_loss_list, val_epoch_loss_list, iter_list, save_path):
    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    ax = [[ax1, ax2], [ax3, ax4], [ax5, ax6]]

    train_label_list = ["train_loss", "train_loss1", "train_loss1"]
    val_label_list = ["val_loss", "val_loss1", "val_loss2"]
    marker_list = ["kx-", "r.-", "b.-"]
    max_loss = float("-inf")
    for i, loss in enumerate(train_epoch_loss_list):
        ax[i][0].plot(iter_list, loss, marker_list[i], label=train_label_list[i])
        if max_loss < max(loss):
            max_loss = max(loss)

    for i, loss in enumerate(val_epoch_loss_list):
        ax[i][1].plot(iter_list, loss, marker_list[i], label=val_label_list[i])
        if max_loss < max(loss):
            max_loss = max(loss)

    ax[0][0].set_title("Training Loss")
    ax[0][1].set_title("Validation Loss")

    ax[0][0].set_ylabel("loss")
    ax[1][0].set_ylabel("loss1")
    ax[2][0].set_ylabel("loss2")

    ax[2][0].set_xlabel("epoch")
    ax[2][1].set_xlabel("epoch")

    ax1.set_yscale("symlog")
    ax2.set_yscale("symlog")
    ax3.set_yscale("symlog")
    ax4.set_yscale("symlog")
    ax5.set_yscale("symlog")
    ax6.set_yscale("symlog")

    for axi in ax:
        for axj in axi:
            axj.set_ylim([0, max_loss])

    figure_path = os.path.join(save_path, "train_val_loss_epoch")
    fig.savefig(figure_path)
