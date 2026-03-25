import argparse
import numpy as np
import pyaudio
import time
import openwakeword
from openwakeword.model import Model

# ================== 配置区 ==================
# 可选：第一次运行时自动下载预训练模型
openwakeword.utils.download_models()

# 支持的预训练唤醒词（你可以改成自己训练的 .onnx 文件）
PRETRAINED_MODELS = ["alexa", "hey_mycroft", "hey_jarvis"]   # 常用几个

# ============================================

parser = argparse.ArgumentParser(description="openWakeWord 实时麦克风唤醒词检测（使用 ONNX）")
parser.add_argument("--inference_framework", type=str, default="onnx",
                    choices=["onnx", "tflite"], help="推理后端（Windows 推荐 onnx）")
parser.add_argument("--model_path", type=str, default=None,
                    help="自定义模型路径（.onnx 文件），留空则加载所有预训练模型")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="检测阈值（0.0~1.0，越大越严格）")
args = parser.parse_args()

# ================== 加载模型 ==================
if args.model_path:
    print(f"加载自定义模型: {args.model_path}")
    owwModel = Model(
        wakeword_models=[args.model_path],
        inference_framework=args.inference_framework
    )
else:
    print("加载所有预训练模型（使用 ONNX）...")
    owwModel = Model(
        inference_framework=args.inference_framework,
        # threshold=args.threshold   # 如果想全局设置阈值可以加
    )

print(f"✅ 成功加载 {len(owwModel.models)} 个唤醒词模型")

# ================== 实时麦克风检测 ==================
CHUNK = 1280          # openWakeWord 推荐的帧长
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("\n🎤 正在监听麦克风... 说唤醒词试试！（按 Ctrl+C 退出）\n")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        # 预测
        prediction = owwModel.predict(audio_frame)

        # 显示检测结果
        for wakeword, score in prediction.items():
            if score >= args.threshold:
                print(f"🚀 唤醒词触发！【{wakeword}】 分数: {score:.4f}   时间: {time.strftime('%H:%M:%S')}")
            elif score > 0.1:   # 只显示有一定置信度的，减少输出
                print(f"   {wakeword:15} → {score:.4f}")

except KeyboardInterrupt:
    print("\n\n👋 检测已停止")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()