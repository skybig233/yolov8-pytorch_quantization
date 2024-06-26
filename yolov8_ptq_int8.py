import argparse
import os
import sys
from pathlib import Path
import warnings
import yaml
import torch
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]          # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO
from utils.dataloaders import create_dataloader
from utils.general import (check_img_size, check_yaml, file_size, colorstr, check_dataset)
from utils.torch_utils import select_device
import py_quant_utils as quant


def collect_stats(model, data_loader, num_batches, device):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, targets, paths, shapes) in tqdm(enumerate(data_loader), total=num_batches):
        image = image.to(device, non_blocking=True)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        model(image)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, quant.calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: detection model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(out_dir, F"{model_name}-max-{num_calib_batch * data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(out_dir, F"{model_name}-percentile-{percentile}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(out_dir, F"{model_name}-{method}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)


def load_model(weight, device):
    yolo = YOLO(weight)
    model = yolo.model
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model,yolo


def prepare_model(calibrator, opt, device):
    
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  
        # print(data_dict)
        data_dict = check_dataset(data_dict)
    calib_path = data_dict['val']

    quant.initialize_calib_method(per_channel_quantization=True, calib_method=calibrator)  
    model,yolo = load_model(opt.weights, device)
    quant.replace_to_quantization_module(model, ignore_policy=opt.sensitive_layer)

    model.eval()
    model.cuda()

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz, _ = [check_img_size(x, gs) for x in [opt.imgsz, opt.imgsz]]  # verify imgsz are gs-multiples

    # Calib dataloader
    calib_loader = create_dataloader(calib_path,
                                     imgsz,
                                     opt.batch_size,
                                     gs,
                                     hyp=None,
                                     cache=opt.cache,
                                     rect=True,
                                     rank=-1,
                                     workers=opt.workers * 2,
                                     pad=0.5,
                                     prefix=colorstr('calib: '))[0]

    return model, calib_loader,yolo


def export_onnx(model, onnx_filename, batch_onnx, dynamic_shape, simplify, imgsz=640, prefix=colorstr('calib: ')):

    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant.quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, imgsz, imgsz)  

    try:
        import onnx
        with torch.no_grad():
            torch.onnx.export(model.cpu(), 
                            dummy_input.cpu(), 
                            onnx_filename, 
                            verbose=False, 
                            opset_version=13, 
                            input_names=['images'],
                            output_names=['output'],
                            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}} if dynamic_shape else None,
                            # enable_onnx_checker=False, 
                            do_constant_folding=True)

        print('ONNX export success, saved as %s' % onnx_filename)

    except ValueError:
        warnings.warn(UserWarning("Per-channel quantization is not yet supported in Pytorch/ONNX RT (requires ONNX opset 13)"))
        print("Failed to export to ONNX")
        return False

    except Exception as e:
            print(f'{prefix} export failure: {e}')
    
    # Checks
    model_onnx = onnx.load(onnx_filename)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    
    # Simplify
    if simplify:
        try:
            import onnxsim
            print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic_shape,
                    input_shapes={'images': list(dummy_input.shape)} if dynamic_shape else None)

            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_filename)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')

        print(f'{prefix} export success, saved as {onnx_filename} ({file_size(onnx_filename):.1f} MB)')
        print(f"{prefix} Run ONNX model inference with: 'python detect.py --weights {onnx_filename}'")
        
    # Restore the PSX/TensorRT's fake quant mechanism
    quant.quant_nn.TensorQuantizer.use_fb_fake_quant = False
    # Restore the model to train/test mode, use Detect() layer grid
    model.export = False

    return True


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '../ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / "../ultralytics/yolov8n.pt", help='model.pt path(s)')
    parser.add_argument('--model-name', '-m', default='yolov8n', help='model name: default yolov5s')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')

    # setting for calibration
    parser.add_argument('--calib-batch-size', type=int, default=32, help='calib batch size: default 64')
    # parser.add_argument('--sensitive-layer', default=[], help='skip sensitive layer: default detect head')

    parser.add_argument('--sensitive-layer', default=['model.15.cv1.conv',
                                                      'model.15.cv2.conv',
                                                      "model.15.m.0.cv1.conv",
                                                      "model.15.m.0.cv2.conv"], help='skip sensitive layer: default detect head')

    parser.add_argument('--num-calib-batch', default=32, type=int,
                        help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--simplify', default=True, help='simplify ONNX file')
    parser.add_argument('--out-dir', '-o', default=ROOT / 'weights/', help='output folder: default ./runs/finetune')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    # print_args(vars(opt))
    return opt




if __name__ == "__main__":

    opt = parse_opt()
    device = select_device(opt.device, batch_size=opt.batch_size)

    model, data_loader, yolo = prepare_model(calibrator=opt.calibrator, opt=opt, device=device)

    # 校准模型
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name=opt.model_name,
            data_loader=data_loader,
            num_calib_batch=opt.num_calib_batch,
            calibrator=opt.calibrator,
            hist_percentile=opt.percentile,
            out_dir=opt.out_dir,
            device=device)


    res = yolo.val(data = opt.data)

    with quant.disable_quantization(model):
        res = yolo.val(data = opt.data)

    onnx_filename = 'yolov8n_ptq_detect.onnx'
    export_onnx(yolo.model, onnx_filename, opt.batch_size_onnx, opt.dynamic, opt.simplify)
