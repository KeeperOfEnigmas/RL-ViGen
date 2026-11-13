import pandas as pd
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_OUTPUT_DIR = os.path.join(ROOT_DIR, "result", "evaluation")

def evaluation(algorithm: tuple, task_list: tuple, seed_list: tuple, evaluation_type: tuple, augmentation: tuple):
    log_path = os.path.join(EVALUATION_OUTPUT_DIR, "error.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Evaluation log started.\n")

    dataframes = {}
    all_html = ""
    for eval_type in evaluation_type:
        results = []
        for algo in algorithm:
            for task in task_list:
                task_rewards = {aug: [] for aug in augmentation}
                for seed in seed_list:
                    dir = os.path.join(ROOT_DIR, "exp_local", algo, task, str(seed), "evaluation", eval_type)
                    try:
                        folders = os.listdir(dir)
                    except FileNotFoundError:
                        log_exception(f"File not found: {dir}")
                        continue

                    for folder in folders:
                        aug = next((aug for aug in augmentation if f"+aug={aug}" in folder), None)
                        if aug is None:
                            continue

                        eval_path = os.path.join(dir, folder, "eval.csv")
                        try:
                            df = load_eval_data(eval_path)
                            rewards = pd.to_numeric(df["episode_reward"], errors="coerce")
                            rewards = rewards[np.isfinite(rewards)]
                            if rewards.empty:
                                log_exception(f"No valid episode_reward values: {eval_path}")
                                continue
                            task_rewards[aug].append(float(rewards.iloc[-1]))
                        except FileNotFoundError:
                            log_exception(f"File not found: {eval_path}")
                        except pd.errors.EmptyDataError:
                            log_exception(f"File is empty: {eval_path}")
                        except Exception as e:
                            log_exception(f"An error occurred while loading the file: {eval_path}\n{str(e)}")

                # --- UPDATED LOGIC ---
                # Instead of storing just the mean, we store the full list of seed rewards
                # to calculate standard deviation later.
                result = {"algorithm": algo, "task": task}
                for aug in augmentation:
                    if eval_type == aug:
                        result[aug] = None
                    else:
                        # Store the list for Std Dev calculation in the exporter
                        result[aug] = task_rewards[aug] 
                results.append(result)

        dataframe = pd.DataFrame(results)
        dataframes[eval_type] = dataframe
        dataframe.set_index(["algorithm", "task"], inplace=True)
        dataframe.columns.name = eval_type.upper()
        
        # We create a display version for CSV/HTML that uses simple means
        display_df = dataframe.copy()
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan)
        
        display_df = display_df.round(2)
        display_df.to_csv(os.path.join(EVALUATION_OUTPUT_DIR, f"{eval_type}.csv"))

        # HTML generation using the mean values
        aug_cols = [aug for aug in augmentation if aug in display_df.columns]
        styled_df = display_df.style.apply(highlight_max, subset=aug_cols, axis=1)
        styled_html = styled_df.to_html()
        
        with open(os.path.join(EVALUATION_OUTPUT_DIR, f"{eval_type}.html"), "w", encoding="utf-8") as f:
            f.write(styled_html)
        all_html += styled_html   
        all_html += '<br>'   

    with open(os.path.join(EVALUATION_OUTPUT_DIR, "all_evaluations.html"), "w", encoding="utf-8") as f:
        f.write(all_html)

    tasks_groups = {
        "locomotion": ["walker_walk", "pendulum_swingup", "humanoid_walk", "cheetah_run"], 
        "robotic": ["Door", "Lift", "TwoArmLift"]
    }
    
    # The dataframe passed here now contains lists of rewards per cell
    save_as_latex_and_txt(dataframes, augmentation, tasks_groups)
    return dataframes


def save_as_latex_and_txt(dataframes, augmentation, task_groups):
    latex_output_dir = os.path.join(EVALUATION_OUTPUT_DIR, "latex_txt")
    os.makedirs(latex_output_dir, exist_ok=True)
    
    for eval_type, df in dataframes.items():
        for group_name, tasks_in_group in task_groups.items():
            available_tasks = [t for t in tasks_in_group if t in df.index.get_level_values('task')]
            if not available_tasks:
                continue
                
            sub_df = df.iloc[df.index.get_level_values('task').isin(available_tasks)]
            algorithms = sub_df.index.get_level_values('algorithm').unique()
            
            # --- HEADER GENERATION ---
            header1 = f"{eval_type.replace('_', ' ').title()} " + ("& " * (len(algorithms) * len(available_tasks))) + "\\\\"
            algo_line = "Algorithms "
            for algo in algorithms:
                algo_line += f"& {algo.upper()} " + ("& " * (len(available_tasks) - 1))
            algo_line += "\\\\"
            
            task_line = "Tasks Augmentations "
            for algo in algorithms:
                for task in available_tasks:
                    task_line += f"& {task.replace('_', ' ')} "
            task_line += "\\\\"
            
            # --- FIND MAX VALUES PER COLUMN ---
            # We pre-calculate max means for each (algorithm, task) column
            max_means = {}
            for algo in algorithms:
                for task in available_tasks:
                    means = []
                    for aug in augmentation:
                        try:
                            rewards = sub_df.loc[(algo, task), aug]
                            if isinstance(rewards, list) and len(rewards) > 0:
                                means.append(np.mean(rewards))
                        except KeyError:
                            continue
                    if means:
                        max_means[(algo, task)] = max(means)

            # --- BODY GENERATION ---
            body_rows = []
            for aug in augmentation:
                latex_aug = aug.replace('_', r'\_')
                row_str = f"{latex_aug:<15}" 
                
                for algo in algorithms:
                    for task in available_tasks:
                        try:
                            rewards = sub_df.loc[(algo, task), aug]
                            if isinstance(rewards, list) and len(rewards) > 0:
                                current_mean = np.mean(rewards)
                                std_val = np.std(rewards)
                                
                                # Formatting decimals
                                # precision = 2 if ("humanoid_walk" in task or "Lift" in task or "Door" in task) else 0
                                precision_mean = 0 if current_mean>=10 else 2
                                precision_std = 0 if std_val>=10 else 2
                                val_core = f"{current_mean:.{precision_mean}f} $\\pm$ {std_val:.{precision_std}f}"
                                
                                # Check if this is the max mean in the column
                                if current_mean == max_means.get((algo, task)):
                                    val_str = f"\\textbf{{{val_core}}}"
                                else:
                                    val_str = val_core
                            else:
                                val_str = ""
                        except KeyError:
                            val_str = ""
                        row_str += f" & {val_str}"
                row_str += " \\\\"
                body_rows.append(row_str)
            
            # --- SAVE FILES ---
            full_content = "\n".join([header1, algo_line, task_line] + body_rows)
            file_base = os.path.join(latex_output_dir, f"{eval_type}_{group_name}")
            
            with open(f"{file_base}.txt", "w", encoding="utf-8") as f:
                f.write(full_content)
                
            with open(f"{file_base}.tex", "w", encoding="utf-8") as f:
                f.write("\\begin{tabular}{l" + "c" * (len(algorithms) * len(available_tasks)) + "}\n")
                f.write(full_content + "\n")
                f.write("\\end{tabular}")


def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]


def load_eval_data(path):
    """
    Load evaluation data from a CSV file.
    Returns a DataFrame with the data.
    """
    df = pd.read_csv(path)
    return df
    

def log_exception(message, log_path=os.path.join(EVALUATION_OUTPUT_DIR, "error.log")):
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


if __name__ == "__main__":
    # Example usage
    algorithm = ("svea", "pieg", )
    # algorithm = ("pieg", )
    task_list = ("walker_walk", "pendulum_swingup", "humanoid_walk", "cheetah_run")
    # task_list = ("Door", "Lift", )
    seed_list = (1, 2, 3, 4, 5, )
    # seed_list = (1, 2, 3, )
    evaluation_type = ("color_easy", "color_hard", "video_easy", "video_hard", "vignette", "distortion", "cutmix", "cutout", "overlay", "cropping", "window", "rotation", "flip_h", "flip_v", "convolution", )
    # evaluation_type = ("vignette", "distortion", "cutmix", "cutout", "overlay", "cropping", "window", "rotation", "flip_h", "flip_v", "convolution", "eval-easy", "eval-medium", "eval-hard", )
    augmentation = ("cutmix", "cutout", "no_aug", "overlay", "cropping", "window", "rotation", "flip_v", "flip_h", "convolution", "mix", )

    evaluation(algorithm, task_list, seed_list, evaluation_type, augmentation)