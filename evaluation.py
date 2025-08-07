import pandas as pd
import os

def evaluation(algorithm: tuple, task_list: tuple, seed_list: tuple, evaluation_type: tuple, augmentation: tuple):
    results = {aug: [] for aug in augmentation}

    for algo in algorithm:
        for task in task_list:
            task_rewards = {aug: [] for aug in augmentation}
            for seed in seed_list:
                for type in evaluation_type:
                    dir = f"exp_local/{algo}/{task}/{seed}/evaluation/{type}/"
                    for folder in os.listdir(dir):
                        for aug in augmentation:
                            if aug in folder:
                                df = load_eval_data(os.path.join(dir, f"{folder}/eval.csv"))
                                reward = df["episode_reward"]
                                task_rewards[aug].append(reward)
            
            # Aggregate across seeds (mean)
            for aug in augmentation:
                results[aug].append(pd.Series(task_rewards[aug]).mean())

    dataframe = pd.DataFrame(results, index=task_list)
    dataframe.columns.name = type.upper()
    print(dataframe)
    return dataframe


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
    algorithm = ("svea",)
    task_list = ("walker_walk",)
    seed_list = (1,)
    evaluation_type = ("color_easy",)
    augmentation = ("cutmix", "cutout", "default", "cropping", "window", "rotation", "flip_v", "flip_h")

    evaluation(algorithm, task_list, seed_list, evaluation_type, augmentation)