import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

X_AXIS = "episode"
Y_AXIS = "episode_reward"



def import_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    data = pd.read_csv(file_path)
    return data

def import_data(algo: str, method):
    training_data_path = f"result/{algo}/{method}/train.csv"
    evaluation_data_path = f"result/{algo}/{method}/eval.csv"

    training_data = import_csv(training_data_path) # buffer_size,episode,episode_length,episode_reward,fps,frame,step,total_time
    evaluation_data = import_csv(evaluation_data_path) # episode,episode_length,episode_reward,frame,step,total_time    

    return training_data, evaluation_data


def visualize(algo: str, ax, method="base", title="Sample"):
    training_data, evaluation_data = import_data(algo, method)

    # ax[0].plot(training_data[X_AXIS], training_data[Y_AXIS], linestyle='None', marker='.', label='Training ' + method)
    train_ma = training_data[Y_AXIS].rolling(window=100, min_periods=1).mean()
    ax[0].plot(training_data[X_AXIS], train_ma, label='Training ' + method)
    ax[0].set_title("Training, \n" + title)
    ax[0].set_xlabel(X_AXIS)
    ax[0].set_ylabel(Y_AXIS)
    ax[0].grid()
    ax[0].legend()

    # ax[1].plot(evaluation_data[X_AXIS], evaluation_data[Y_AXIS], linestyle='None', marker='.', label='Evaluation ' + method)
    eval_ma = evaluation_data[Y_AXIS].rolling(window=20, min_periods=1).mean()
    ax[1].plot(evaluation_data[X_AXIS], eval_ma, label='Evaluation ' + method)
    ax[1].set_title("Evaluation, \n" + title)
    ax[1].set_xlabel(X_AXIS)
    ax[1].set_ylabel(Y_AXIS)
    ax[1].grid()
    ax[1].legend()

    return ax


def compare(algo: str, ax, method: str=None):
    if method is not None:
        training_data, evaluation_data = import_data(algo, method)

        # ax[0].plot(training_data[X_AXIS], training_data[Y_AXIS], linestyle='None', marker='.', label='Training ' + method)
        train_ma = training_data[Y_AXIS].rolling(window=100, min_periods=1).mean()
        ax[0].plot(training_data[X_AXIS], train_ma, label='Training ' + method)
        ax[0].set_title(ax[0].get_title() + f", {method}")
        ax[0].legend()

        # ax[1].plot(evaluation_data[X_AXIS], evaluation_data[Y_AXIS], linestyle='None', marker='.', label='Evaluation ' + method)
        eval_ma = evaluation_data[Y_AXIS].rolling(window=20, min_periods=1).mean()
        ax[1].plot(evaluation_data[X_AXIS], eval_ma, label='Evaluation ' + method)
        ax[1].set_title(ax[1].get_title() + f", {method}")
        ax[1].legend()
    else:
        raise ValueError(f"Method can't be None!")



if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    #  visualize("svea", ax, method="random_crop_slight_color", title="SVEA, walker_walk, random_crop_slight_color")
    visualize("svea", axs, method="base_slight_color", title="SVEA, walker_walk, base")
    compare("svea", axs, method="base_no_augmentation")
    compare("svea", axs, method="random_crop_slight_color")
    compare("svea", axs, method="random_rot_slight_color")
    compare("svea", axs, method="random_window_slight_color")
    compare("svea", axs, method="random_flip_h_slight_color")
    compare("svea", axs, method="random_flip_v_slight_color")
    compare("svea", axs, method="random_cutout_slight_color")
    compare("svea", axs, method="random_conv_slight_color")
    compare("svea", axs, method="random_cutmix_slight_color")
    plt.show()