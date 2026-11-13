import matplotlib.pyplot as plt
import pandas as pd
import os


PHOTOMETRIC = ["overlay", "convolution"]
GEOMETRIC = ["cropping", "window", "rotation", "flip_v", "flip_h", "cutout", "cutmix"]
BASELINE = ["no_aug"]
MIX = ["mix"]



def import_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    data = pd.read_csv(file_path)

    return data


def sort_df(file_path):
    df = import_csv(file_path)

    sub_dfs = {}
    for (algo, task), sub_df in df.groupby(["algorithm", "task"]):
        sub_dfs[(algo, task, "photometric")] = sub_df[["algorithm", "task"] + PHOTOMETRIC]
        sub_dfs[(algo, task, "geometric")]   = sub_df[["algorithm", "task"] + GEOMETRIC]
        sub_dfs[(algo, task, "baseline")]    = sub_df[["algorithm", "task"] + BASELINE]
        sub_dfs[(algo, task, "mix")]         = sub_df[["algorithm", "task"] + MIX]

    print(sub_dfs)
    return sub_dfs


def visualize(file_path, algorithm, task):
    sorted = sort_df(file_path)
    sub_df_geometric = sorted[(algorithm, task, "geometric")]
    sub_df_photometric = sorted[(algorithm, task, "photometric")]
    sub_df_baseline = sorted[(algorithm, task, "baseline")]
    sub_df_mix = sorted[(algorithm, task, "mix")]

    # Extract augmentation values, assuming one row per df
    row_geometric = sub_df_geometric.iloc[0]
    row_photometric = sub_df_photometric.iloc[0]
    row_baseline = sub_df_baseline.iloc[0]
    row_mix = sub_df_mix.iloc[0]

    # Get scores
    geometric_mean = row_geometric[GEOMETRIC].dropna().mean()
    photometric_mean = row_photometric[PHOTOMETRIC].dropna().mean()
    baseline_mean = row_baseline[BASELINE].dropna().mean()
    mix_mean = row_mix[MIX].dropna().mean()

    groups = ["Geometric", "Photometric","No Aug", "Mix"]
    scores = [geometric_mean, photometric_mean, baseline_mean, mix_mean]

    # Plot
    plt.bar(groups, scores)
    plt.title(f"Color Easy")
    plt.xlabel("Augmentation")
    plt.ylabel("Score")
    plt.show()



if __name__ == "__main__":
    visualize("result/evaluation/color_easy.csv", "svea", "pendulum_swingup")