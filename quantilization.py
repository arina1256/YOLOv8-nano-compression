"""
INT8 КВАНТОВАНИЕ через TensorRT (требует NVIDIA GPU)
"""

from ultralytics import YOLO

MODEL_PATH = '/content/best.pt'
DATASET_ROOT = '/content/drive/MyDrive/minidataset'

model = YOLO(MODEL_PATH)

model.export(
    format='engine',      
    half=False,          
    int8=True,           
    data=DATASET_ROOT + '/data.yaml',  
    imgsz=640,
    batch=1
)

print("INT8 TensorRT модель создана: best.engine")
