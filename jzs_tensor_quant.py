from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

class NewQuantizer(TensorQuantizer):
    def __init__(self, quant_desc=..., disabled=False, if_quant=True, if_clip=False, if_calib=False):
        super().__init__(quant_desc, disabled, if_quant, if_clip, if_calib)



import torch  
import torch.nn as nn  

# 定义卷积网络  
class SimpleConvNet(nn.Module):  
    def __init__(self):  
        super(SimpleConvNet, self).__init__()  
        self.conv = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)  
        self.relu = nn.ReLU()  
        self.maxpool = nn.MaxPool2d(kernel_size=2)  
  
    def forward(self, x):  
        x = self.conv(x)  
        x = self.relu(x)  
        x = self.maxpool(x)  
        return x  


def tensor_quant_test():
    import torch
    quant_desc = QuantDescriptor(num_bits=8, fake_quant=True, calib_method='histogram')
    quantizer = TensorQuantizer(quant_desc)
    # quantizer.enable_clip()
    torch.manual_seed(12345)
    x = torch.rand(5)

    # x=x.repeat(2,1)
    print(x)
    quant_x = quantizer(x)
    print(quant_x)



def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

if __name__=="__main__":
    input_tensor = torch.randn(1, 1, 64, 64)
    print(input_tensor)
    net=SimpleConvNet()
    output_tensor = net(input_tensor)
    print(output_tensor)

    import py_quant_utils as quant
    quant.qyolo_calib_method()
    quant.replace_to_quantization_module(net)
    quant.quantizer_state(net)

    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=2)
        compute_amax(model, method="percentile", percentile=99.99)
        
    quant_output=net(input_tensor)
    print(quant_output)