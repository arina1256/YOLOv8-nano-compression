import os
from ultralytics import YOLO

fp32_path = '/content/runs/detect/minidataset25/train_run25/weights/best.pt'
int8_path = '/content/runs/detect/minidataset25/train_run25/weights/best.engine'

fp32_size = os.path.getsize(fp32_path) / 1e6
int8_size = os.path.getsize(int8_path) / 1e6

print(f"FP32 модель: {fp32_size:.2f} MB")
print(f"INT8 модель: {int8_size:.2f} MB")
print(f"Сжатие: {(1 - int8_size/fp32_size)*100:.1f}%")

if os.path.exists(int8_path):
    model_int8 = YOLO(int8_path)
    print("INT8 модель успешно загружена!")
