from ultralytics import YOLO
import py_quant_utils as quant
weight= "weights/yolov8n-mse-1024.pth"
data='../ultralytics/ultralytics/cfg/datasets/coco.yaml'
yolo = YOLO(weight)
# model=yolo.model
# with quant.enable_quantization(model):
#     print("after quantization")
res = yolo.val(data = data)