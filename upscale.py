import cv2
from cv2 import dnn_superres


input_video = r"D:\Git\RL-ViGen\upscale\color_easy.mp4"
output_path = r"D:\Git\RL-ViGen\upscale\output_color_easy.mp4"

# 1. 初始化超分辨率模型（视频推荐 ESPCN 或 FSRCNN）
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(r"D:\Git\RL-ViGen\upscale\model\FSRCNN_x2.pb")
sr.setModel("fsrcnn", 2)

# (推荐) 开启 CUDA GPU 加速，大幅提升视频处理速度
# sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 2. 打开输入视频并读取属性
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise ValueError("无法打开视频文件")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3. 计算输出视频分辨率并初始化 VideoWriter
scale = 2  # 必须与模型倍率一致
out_w, out_h = orig_w * scale, orig_h * scale

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

print(f"开始处理: 原分辨率 ({orig_w}x{orig_h}) -> 放大后 ({out_w}x{out_h})，总帧数: {total_frames}")

# 4. 逐帧处理
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 逐帧超分辨率重建
    upsampled_frame = sr.upsample(frame)

    # 写入新视频
    out.write(upsampled_frame)

    frame_count += 1
    if frame_count % 30 == 0 or frame_count == total_frames:
        print(f"处理进度: {frame_count}/{total_frames} 帧 ({(frame_count / total_frames) * 100:.1f}%)")

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print("视频处理完成！")