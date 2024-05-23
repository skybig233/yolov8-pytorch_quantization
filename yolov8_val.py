from ultralytics import YOLO
import py_quant_utils as quant
weight="./yolov8n-obb.pt"
data="../ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml"
yolo = YOLO(weight)
# model = yolo.model
# model.float()
# model.eval()
# with quant.enable_quantization(model):
metrics =yolo.val(data=data)
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category