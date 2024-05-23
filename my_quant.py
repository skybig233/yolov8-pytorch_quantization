
from ultralytics import YOLO
import yaml

from py_quant_utils import quantizer_state
from utils.qan import find_modules_to_quantize, replace_module_by_names

weight="../ultralytics/yolov8n.pt"
yolo = YOLO(weight)
model = yolo.model
# q-model
quan_yaml="/home/jzs/cv/yolov8-pytorch_quantization/config/yolov3-tiny_lsq.yaml"
with open(quan_yaml, 'r') as load_f:
    qcfg = yaml.load(load_f, Loader=yaml.FullLoader)
modules_to_replace = find_modules_to_quantize(model, qcfg)
model = replace_module_by_names(model, modules_to_replace)
quantizer_state(model)