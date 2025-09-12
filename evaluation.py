import pandas as pd
import os

def evaluation(algorithm: tuple, task_list: tuple, seed_list: tuple, evaluation_type: tuple, augmentation: tuple):
    dataframes = {}
    all_html = ""
    for eval_type in evaluation_type:
        results = []
        for algo in algorithm:
            for task in task_list:
                task_rewards = {aug: [] for aug in augmentation}
                for seed in seed_list:
                    dir = f"exp_local/{algo}/{task}/{seed}/evaluation/{eval_type}/"
                    for folder in os.listdir(dir):
                        for aug in augmentation:
                            if f"+aug={aug}" in folder:
                                df = load_eval_data(os.path.join(dir, f"{folder}/eval.csv"))
                                reward = df["episode_reward"].values[0]
                                task_rewards[aug].append(reward)

                # Aggregate across seeds (mean)
                result = {"algorithm": algo, "task": task}
                # result = {"task": task}
                for aug in augmentation:
                    if eval_type==aug:
                        result[aug] = float("nan")
                    else:
                        result[aug] = pd.Series(task_rewards[aug], dtype="float64").mean()
                results.append(result)

        dataframe = pd.DataFrame(results)
        dataframes[eval_type] = dataframe
        dataframe.set_index(["algorithm", "task"], inplace=True)
        dataframe.columns.name = eval_type.upper()
        # dataframe.columns = pd.MultiIndex.from_product([[eval_type.upper()], dataframe.columns])
        aug_cols = [aug for aug in augmentation if aug in dataframe.columns]
        dataframe = dataframe.style.apply(highlight_max, subset=aug_cols, axis=1)
        # print(f"\nDataFrame for {eval_type}:")
        # print(dataframe)
        styled_html = dataframe.to_html()
        with open(f"result/evaluation/{eval_type}.html", "w", encoding="utf-8") as f:
            f.write(styled_html) # Save individual HTML files.
        all_html += styled_html  # Concatenate all HTML strings.   
        all_html += '<br>'   

    with open("result/evaluation/all_evaluations.html", "w", encoding="utf-8") as f:
        f.write(all_html)

    # print(dataframes)
    return dataframes


def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]


def load_eval_data(path):
    """
    Load evaluation data from a CSV file.
    Returns a DataFrame with the data.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file: {path}\n{str(e)}")
    


if __name__ == "__main__":
    # Example usage
    algorithm = ("svea", "pieg", )
    task_list = ("walker_walk", "pendulum_swingup", )
    seed_list = (1, 2, )
    evaluation_type = ("color_easy", "color_hard", "video_easy", "video_hard", "vignette", "distortion", "cutmix", "cutout", "overlay", "cropping", "window", "rotation", "flip_h", "flip_v", "convolution", )
    # augmentation = ("cutmix", "cutout", "overlay", "cropping", "window", "rotation", "flip_v", "flip_h", "convolution", )
    augmentation = ("cutmix", "cutout", "no_aug", "overlay", "cropping", "window", "rotation", "flip_v", "flip_h", "convolution", "mix")

    evaluation(algorithm, task_list, seed_list, evaluation_type, augmentation)