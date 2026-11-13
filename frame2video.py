import cv2
import os
from glob import glob

input_dir = r"D:\Git\RL-ViGen\upscale\realesrgan-ncnn-vulkan-20220424-windows\out_frames"
output_path = r"D:\Git\RL-ViGen\upscale\output_color_easy.mp4"

files = sorted(glob(os.path.join(input_dir, "*.jpg")))

# 读取第一帧确定尺寸
first = cv2.imread(files[0])
height, width = first.shape[:2]

fps = 30  # 根据你的原视频设置

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

for file in files:
    frame = cv2.imread(file)

    if frame is None:
        print(f"Warning: cannot read {file}")
        continue

    writer.write(frame)

writer.release()

print(f"Saved to {output_path}")