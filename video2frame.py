import cv2
import os

input_video = r"D:\Git\RL-ViGen\upscale\color_easy.mp4"
video_path = input_video
output_dir = r"D:\Git\RL-ViGen\upscale\realesrgan-ncnn-vulkan-20220424-windows\tmp_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imwrite(
        os.path.join(output_dir, f"{frame_id:06d}.png"),
        frame
    )

    frame_id += 1

cap.release()

print("FPS:", fps)
print("Frames:", frame_id)

