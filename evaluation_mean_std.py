import pandas as pd
import glob
import os


# ============================================================
# Configuration
# ============================================================

DATA_FOLDER = r"D:\Git\RL-ViGen\result\evaluation\mean_std"

# You can specify one or multiple values.
# All selected values are combined together.

# ALGORITHMS = ["pieg"]
ALGORITHMS = ["svea", "pieg"]

TASKS = ["walker_walk"]
# TASKS = ["walker_walk", "pendulum_swingup"]
# TASKS = ["cheetah_run", "humanoid_walk"]
# TASKS = ["Door", "Lift"]

AUGMENTATIONS = ["flip_h"]

FILE_NAME = ["color_easy", "color_hard"]
# FILE_NAME = ["color_easy", "color_hard","video_easy", "video_hard"]


# ============================================================
# Read all CSV files and collect scores
# ============================================================

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {DATA_FOLDER}"
    )


all_scores = []

for file in csv_files:

    if os.path.splitext(os.path.basename(file))[0] not in FILE_NAME:
        continue

    df = pd.read_csv(file)

    # Check required columns
    if "algorithm" not in df.columns:
        print(f"Skipping {os.path.basename(file)}: "
              "missing 'algorithm' column")
        continue

    if "task" not in df.columns:
        print(f"Skipping {os.path.basename(file)}: "
              "missing 'task' column")
        continue

    # --------------------------------------------------------
    # Select rows matching the specified algorithms and tasks
    # --------------------------------------------------------

    filtered = df[
        df["algorithm"].isin(ALGORITHMS) &
        df["task"].isin(TASKS)
    ]

    if filtered.empty:
        continue

    # --------------------------------------------------------
    # Collect scores from all specified augmentations
    # --------------------------------------------------------

    for augmentation in AUGMENTATIONS:

        if augmentation not in df.columns:
            print(
                f"Warning: '{augmentation}' not found in "
                f"{os.path.basename(file)}"
            )
            continue

        scores = pd.to_numeric(
            filtered[augmentation],
            errors="coerce"
        ).dropna()

        all_scores.extend(scores.tolist())


# ============================================================
# Calculate combined mean and std
# ============================================================

if not all_scores:

    print("No matching scores found.")

else:

    series = pd.Series(all_scores)

    mean = series.mean()
    std = series.std(ddof=1)

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)

    print(f"Algorithms:    {ALGORITHMS}")
    print(f"Tasks:         {TASKS}")
    print(f"Augmentations: {AUGMENTATIONS}")

    print(f"\nN:              {len(all_scores)}")
    print(f"Mean:           {mean:.4f}")
    print(f"Std:            {std:.4f}")

    print("\nAll scores:")
    print(all_scores)