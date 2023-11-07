

from typing import List
import matplotlib.pyplot as plt
from matplotlib import colors
from arc.interface import Riddle, BoardPair

def plot_one(ax, train_or_test,input_or_output,input_matrix):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)

def plot_task(task:Riddle,save=False,path_to_save='./tmp/task.png'):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(task.train)
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):
        plot_one(axs[0,i],'train','input',task.train[i].input.np)
        plot_one(axs[1,i],'train','output',task.train[i].output.np)
    plt.tight_layout()
    plt.show(block = False)

    num_test = len(task.test)
    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
    if num_test==1:
        plot_one(axs[0],'test','input',task.test[0].input.np)
        plot_one(axs[1],'test','output',task.test[0].output.np)
    else:
        for i in range(num_test):
            plot_one(axs[0,i],'test','input',task.test[i].input.np)
            plot_one(axs[1,i],'test','output',task.test[i].output.np)
    plt.tight_layout()
    if save:
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        plt.close('all')
        # save image to ./tmp
        import numpy as np
        import PIL
        X = np.asarray(buf)
        im = PIL.Image.fromarray(X)
        im.save(path_to_save)
    else:
        plt.show(block=False)

def plot_pairs(task:List[BoardPair]):
    """
    Plots pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(task)
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    if num_train==1:
        plot_one(axs[0],'test','input',task.test[0].input.np)
        plot_one(axs[1],'test','output',task.test[0].output.np)
    else:
        for i in range(num_train):
            plot_one(axs[0,i],'test','input',task.test[i].input.np)
            plot_one(axs[1,i],'test','output',task.test[i].output.np)
    plt.tight_layout()
    plt.show(block=False)
