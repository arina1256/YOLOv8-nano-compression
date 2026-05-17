import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import os


MODEL_PATH = '/content/runs/detect/minidataset25/train_run25/weights/best.pt'
PRUNED_PATH = '/content/runs/detect/minidataset25/train_run25/weights/pruned_simple.pt'
DATASET_ROOT = Path('/content/drive/MyDrive/minidataset')


def clone_model_with_pruning(original_model, prune_ratio=0.3):
   
    
    print(f"\n  Применяем прунинг {prune_ratio*100:.0f}%")
    params_before = sum(p.numel() for p in original_model.model.parameters())
    print(f"  Параметров до: {params_before/1e6:.2f}M")
    pruned_model = YOLO(MODEL_PATH)
    pruned_count = 0
    for name, module in pruned_model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.out_channels > 64:  
                old_out = module.out_channels
                new_out = max(32, int(old_out * (1 - prune_ratio)))
                if new_out < old_out:
                    with torch.no_grad():
                        module.weight.data = module.weight.data[:new_out].clone()
                        if module.bias is not None:
                            module.bias.data = module.bias.data[:new_out].clone()
                        module.out_channels = new_out
                    
                    pruned_count += 1
                    if pruned_count <= 20: 
                        print(f"    {name}: {old_out} -> {new_out}")
    
    
    params_after = sum(p.numel() for p in pruned_model.model.parameters())
    reduction = (1 - params_after/params_before) * 100
    
    print(f"\n Результат:")
    print(f"  Параметров после: {params_after/1e6:.2f}M")
    print(f"  Сжатие: {reduction:.1f}%")
    print(f"  Обрезано слоев: {pruned_count}")
    
    return pruned_model


print("ПРУНИНГ МОДЕЛИ")

print(f"\n Загружаем модель: {MODEL_PATH}")
original_model = YOLO(MODEL_PATH)

PRUNE_RATIO = 0.3  
pruned_model = clone_model_with_pruning(original_model, PRUNE_RATIO)

print(f"\n Сохранение обрезанной модели")
pruned_model.save(PRUNED_PATH)
print(f"  Сохранено: {PRUNED_PATH}")

fp32_size = os.path.getsize(MODEL_PATH) / 1e6
pruned_size = os.path.getsize(PRUNED_PATH) / 1e6
print(f"\n  Размер FP32: {fp32_size:.2f} MB")
print(f"  Размер Pruned: {pruned_size:.2f} MB")
print(f"  Сжатие файла: {(1 - pruned_size/fp32_size)*100:.1f}%")


print("ДООБУЧЕНИЕ ОБРЕЗАННОЙ МОДЕЛИ")

pruned_model_for_train = YOLO(PRUNED_PATH)

try:
    results = pruned_model_for_train.train(
        data=str(DATASET_ROOT / 'data.yaml'),
        epochs=20,
        imgsz=640,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=1,
        project='minidataset25',
        name='pruned_finetuned_simple',
        verbose=True
    )
    print("\n Дообучение завершено!")
    
    finetuned_path = '/content/runs/detect/minidataset25/pruned_finetuned_simple/weights/best.pt'
    
except Exception as e:
    print(f" Ошибка дообучения: {e}")
    finetuned_path = None


print("\n" )
print("СРАВНЕНИЕ МОДЕЛЕЙ")

print("\nВалидация FP32 модели")
res_fp32 = original_model.val(data=str(DATASET_ROOT/'data.yaml'), split='test', verbose=False)
print(f"  FP32 mAP50-95: {res_fp32.box.map:.4f}")

if finetuned_path and os.path.exists(finetuned_path):
    print("\nВалидация обрезанной и дообученной модели")
    model_pruned = YOLO(finetuned_path)
    res_pruned = model_pruned.val(data=str(DATASET_ROOT/'data.yaml'), split='test', verbose=False)
    print(f"  Pruned mAP50-95: {res_pruned.box.map:.4f}")
    
    print(f"\n{'Показатель':<20} {'FP32':<12} {'Pruned':<12} {'Изменение':<10}")
    print("-"*55)
    map_change = (res_pruned.box.map - res_fp32.box.map) / res_fp32.box.map * 100
    print(f"{'mAP50-95':<20} {res_fp32.box.map:.4f}       {res_pruned.box.map:.4f}       {map_change:+.2f}%")
    print(f"{'Size (MB)':<20} {fp32_size:.2f}       {pruned_size:.2f}       {(1 - pruned_size/fp32_size)*100:+.1f}%")
else:
    
    print("ИТОГОВАЯ СВОДКА")
    
    params_before = sum(p.numel() for p in original_model.model.parameters()) / 1e6
    params_after = sum(p.numel() for p in pruned_model.model.parameters()) / 1e6
    
    print(f"""
     Результат прунинга:
    
    Параметры:     {params_before:.2f}M → {params_after:.2f}M
    Размер:        {fp32_size:.2f} MB → {pruned_size:.2f} MB
    Сжатие:        {(1 - pruned_size/fp32_size)*100:.1f}%
    
    """)


print("ГОТОВО!")
    