import torch
import torch.nn as nn
import functools
from functools import partial
# import torchvision.models as models
# import torch.ao.quantization
import math
import copy
import torch.nn.functional as F  
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import numpy as np


def auto_quant(arch, model, calib_image_list, precision='INT8', acc_loss=0.01, mode='PTQ', device='cuda', useConv2D=True, useSmooth=True):

    if 'resnet' in arch or 'ResNet' in arch:
        arch = 'resnet50'
    elif 'yolo' in arch or 'YOLO' in arch or 'Yolo' in arch:
        arch = 'yolov2'
    else:
        raise ValueError("Sorry, this network do not support now ") 

    model.eval()

    # return fused_model
    fused_model = fuse_conv_bn(model)
    smooth_input = {}
    if useSmooth == True:
        smoothed_model, input_scale, output_scale, residual_scale, smooth_input = smooth(arch, fused_model, calib_image_list, alpha=0.5)
        print(smoothed_model)
        # print(smoothed_model.named_modules())
        # assert()              # 替换层后model找不到，应该怎么办，明天查
        weight_quant_model, weight_scale, bias_scale = get_weight_scale_and_quant(smoothed_model)
        # print(weight_scale)
    else:
        input_scale, output_scale, residual_scale = get_scales(arch, fused_model, calib_image_list)
        weight_quant_model, weight_scale, bias_scale = get_weight_scale_and_quant(fused_model)
        # print(weight_quant_model)
        # print(input_scale)
        # print(output_scale)
        # print(weight_scale)
        # print(bias_scale)
        # print("output")
        # for name, scale in output_scale.items():
        #     print(name, scale)
        # print("weight")
        # for name, scale in weight_scale.items():
        #     print(name, scale)
        # assert()
    
    # weight_quant_model, weight_scale, bias_scale = get_weight_scale_and_quant(fused_model)
    
    # input_max_scale = get_input_scale(fused_model, calib_image_list)
    if arch == 'resnet50':
        # input_max_scale = get_input_scale(fused_model, calib_image_list)
        input_max_scale = input_scale['conv1']
        input_scale = update_scale(weight_scale, input_scale, output_scale, input_max_scale)
    elif arch =='yolov2':
        input_max_scale = input_scale['backbone.conv_1.0.convs.0']
    
    

    final_scale = 1
    for key in residual_scale:
        final_scale = residual_scale[key]
    quantized_model, act_weight_scale = quant_module(arch, weight_quant_model, weight_scale, input_scale, 
                                                     output_scale, residual_scale, input_max_scale, final_scale, bias_scale, smooth_input, useConv2D, useSmooth)
    quantized_model = quantized_model.to(device)

    # test_yolov2(arch, quantized_model, calib_image_list, input_scale, output_scale)
    # assert()
    # optimized_model, optimized_scale = actscale_optimize(quantized_model, input_max_scale, weight_scale, act_weight_scale)

    return quantized_model



def test_yolov2(arch, model, calib_list, input_scale, output_scale):
    model.eval()
    device = next(model.parameters()).device
    input_max = {}
    output_max = {}
    intput_min = {}
    output_min = {}
    residual_max = {}
    
    # input_dict = {}
    # output_dict = {}
    # residual_dict ={}

    def hook_function(module, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
        if not isinstance(input, torch.Tensor) or not isinstance(output, torch.Tensor):  
            return
        # print(name)
        # print(output.shape)
        if arch == 'resnet50':
            max_input_range = max(torch.max(input), abs(torch.min(input))).item()
            max_output_range = max(torch.max(output), abs(torch.min(output))).item()

        elif arch =='yolov2':
            # if isinstance(input, torch.Tensor):  
                max_input_range = torch.max(input).item()
            # if isinstance(output, torch.Tensor):  
                max_output_range = torch.max(output).item()

        if name in input_max:
            input_max[name] = max(input_max[name], max_input_range)
        else:
            input_max[name] = max_input_range
        if name in output_max:
            output_max[name] = max(output_max[name], max_output_range)
        else:
            output_max[name] = max_output_range
        # min_input = torch.min(input).item()
        # min_output = torch.min(output).item()
        # if name in intput_min:
        #     intput_min[name] = min(min_input, intput_min[name])
        # else:
        #     intput_min[name] = min_input
        # if name in output_min:
        #     output_min[name] = min(min_output, output_min[name])
        # else:
        #     output_min[name] = min_output
        
        # if name not in input_dict:
        #     input_dict[name] = []
        # input_dict[name].append(input.cpu())     # many GPU memory used
        # if name not in output_dict:
        #     output_dict[name] = []
        # output_dict[name].append(output.cpu())
        
        # if name not in residual_dict:
        #     residual_dict[name] = []
        # residual_dict[name].append(output.cpu())
        
    
    hooks = []
    for name, m in model.named_modules():
        hooks.append(
                m.register_forward_hook(functools.partial(hook_function, name=name))
            )

    if arch == 'resnet50':
        with torch.no_grad():
            for i, (images, target) in enumerate(calib_list):
                
                images = images.to(device)
                output = model(images)
    elif arch =='yolov2':
        num_images = len(calib_list)

        for i in range(num_images):
            im, gt, h, w = calib_list.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(device)
            # forward
            bboxes, scores, cls_inds = model(x)
            break
            if i == 100:
                break


    for h in hooks:
        h.remove()
    # for h in residual_hooks:
    #     h.remove()
    

    # input_scale = {}
    # output_scale = {}
    # residual_scale = {}

    print("Start!")
    print("Input!")
        # input_scale[name] = MSE_update_act(input_dict[name], input_scale[name])
    for name, max_range in input_max.items():
        # output_scale[name] = max_range*2/254
        print(name)
        print(max_range)
        if name in input_scale:
            print("the resume is :")
            print(max_range*input_scale[name])
    print("Output!")
    for name, max_range in output_max.items():
        # output_scale[name] = max_range*2/254
        print(name)
        print(max_range)
        if name in output_scale:
            print("the resume is :")
            print(max_range*output_scale[name])
        # output_scale[name] = MSE_update_act(output_dict[name], output_scale[name])
    
    return

def smooth_conv2d(model, name):
    for child_name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
                string = name+'.'+child_name
                string = string[1:]
                weight_string = string
                model._modules[child_name] = SmoothConv2d(module)
        else:
            smooth_conv2d(module, name+'.'+child_name)






def smooth(arch, model, calib_list, alpha=0.5):
    model.eval()
    device = next(model.parameters()).device
    input_max = {}
    output_max = {}
    intput_min = {}
    output_min = {}
    residual_max = {}
    
    # input_dict = {}
    # output_dict = {}
    # residual_dict ={}
    input_col_max = {}
    weight_row_max = {}

    def hook_function(module, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
        
        if isinstance(module, nn.Conv2d):
        # if 'conv' in name:

            col = im2col(input, module.weight.shape[2], module.weight.shape[3], module.stride, module.padding)
            # col = torch.from_numpy(col).float()
            col_W = module.weight.reshape(module.weight.shape[0], -1).T
            # print(input.size())
            # print(col.size())
            # print(col_W.size())

            max_values_per_column, _ = torch.max(col, dim=0, keepdim=False)  
            min_values_per_column, _ = torch.min(col, dim=0, keepdim=False)  
            max_values_per_row, _ = torch.max(col_W, dim=1, keepdim=False)
            min_values_per_row, _ = torch.min(col_W, dim=1, keepdim=False)
            max_act = torch.maximum(max_values_per_column, abs(min_values_per_column))
            max_weight = torch.maximum(max_values_per_row, abs(min_values_per_row))

            if name in input_col_max:
                input_col_max[name] = torch.maximum(input_col_max[name], max_act)
            else:
                input_col_max[name] = max_act
            if name in weight_row_max:
                weight_row_max[name] = torch.maximum(weight_row_max[name], max_weight)
            else:
                weight_row_max[name] = max_weight
            # print(len(max_values_per_row))
            # print(max_values_per_row)
            # print(len(max_values_per_column))
            # print(max_values_per_column)


        max_input_range = max(torch.max(input), abs(torch.min(input))).item()
        max_output_range = max(torch.max(output), abs(torch.min(output))).item()
        if name in input_max:
            input_max[name] = max(input_max[name], max_input_range)
        else:
            input_max[name] = max_input_range
        if name in output_max:
            output_max[name] = max(output_max[name], max_output_range)
        else:
            output_max[name] = max_output_range
        # min_input = torch.min(input).item()
        # min_output = torch.min(output).item()
        # if name in intput_min:
        #     intput_min[name] = min(min_input, intput_min[name])
        # else:
        #     intput_min[name] = min_input
        # if name in output_min:
        #     output_min[name] = min(min_output, output_min[name])
        # else:
        #     output_min[name] = min_output
        # print(name)
        # print(input.size())
        # print(module.weight.size())
        # print(output.size())
        

    def hook_residual(module, input, output, name):
        max_value = torch.max(output).item()
        if name in residual_max:
            residual_max[name] = max(max_value, residual_max[name])
        else:
            residual_max[name] = max_value
        
        # if name not in residual_dict:
        #     residual_dict[name] = []
        # residual_dict[name].append(output.cpu())
        
    
    hooks = []
    residual_hooks = []
    for name, m in model.named_modules():
        hooks.append(
                m.register_forward_hook(functools.partial(hook_function, name=name))
            )
        if 'Bottleneck' in str(type(m)):
            residual_hooks.append(
                m.register_forward_hook(functools.partial(hook_residual, name=name))
            )

    with torch.no_grad():
        for i, (images, target) in enumerate(calib_list):
            
            images = images.to(device)
            output = model(images)
            break


    for h in hooks:
        h.remove()
    for h in residual_hooks:
        h.remove()
    

    smooth_scale = {}
    
    for name, max_range in input_col_max.items():
        print(name)
        # print(max_range)
        smooth_scale[name] = pow(input_col_max[name], alpha) / pow(weight_row_max[name], 1 - alpha)
        # print(smooth_scale[name])
        smooth_scale[name] = smooth_scale[name].clamp(min = 1e-8)
        # print(max_range/smooth_scale[name])
        input_max[name] = torch.max(max_range/smooth_scale[name]).item()
        print(input_max[name])
    # assert()

    # new_conv_layer = {} 
    for name, module in model.named_modules():
        if name in smooth_scale:
            smoothed = SmoothConv2d(module, weight_shape=module.weight.shape, input_smooth=smooth_scale[name])
            smoothed.weight = Parameter(module.weight.reshape(module.weight.shape[0], -1).T * smooth_scale[name].view(-1,1))
            # new_conv_layer[name] = smoothed
            # setattr(model, name, new_conv_layer)
            _set_module(model, name, smoothed)
    # for name, scale in smooth_scale.items():
    #     module = model._modules[name]
    #     smoothed = SmoothConv2d(module, weight_shape=module.weight.shape)
    #     smoothed.weight = module.weight.reshape(module.weight.shape[0], -1).T * smooth_scale[name].view(-1,1)
    #     model._modules[name] = smoothed
    # smooth_model = smooth_weight(model, smooth_scale)

    input_scale = {}
    output_scale = {}
    residual_scale = {}

    print("Here!")
    for name, max_range in input_max.items():
        input_scale[name] = max_range*2/254
        # input_scale[name] = MSE_update_act(input_dict[name], input_scale[name])
    for name, max_range in output_max.items():
        output_scale[name] = max_range*2/254
        # output_scale[name] = MSE_update_act(output_dict[name], output_scale[name])
    for name, max_range in residual_max.items():
        residual_scale[name] = max_range*2/254
        # residual_scale[name] = MSE_update_act(residual_dict[name], residual_scale[name])
    
    return model, input_scale, output_scale, residual_scale, smooth_scale


# def smooth_weight(model, smooth_scale):
#     return model
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)



class SmoothConv2d(nn.Module):
    # def __init__(self, conv_layer, weight_slice=1, input_slice=1, output_slice=1, bias_slice=1, input_max_scale=1):
    def __init__(self, conv_layer, weight_slice=1, input_slice=1, output_slice=1, bias_slice=1, input_max_scale=1, input_smooth=1, weight_shape=[1,1,1,1]):
        super(SmoothConv2d, self).__init__()  

        self.stride = conv_layer.stride
        self.kernel_size = conv_layer.kernel_size
        self.padding = conv_layer.padding
        self.weight_slice = weight_slice
        self.input_slice = input_slice
        self.output_slice = output_slice
        self.bias_slice = bias_slice
        self.input_max_scale = input_max_scale
        self.input_smooth = input_smooth
        self.shape = weight_shape

        self.weight = Parameter(conv_layer.weight.data)
        self.bias = Parameter(conv_layer.bias.data)
        # self.weight = conv_layer.weight.data
        # self.bias = conv_layer.bias.data
        
    def forward(self, input_data):
        if self.input_max_scale != 1:
            s = 1/self.input_max_scale
            input_data = torch.round(input_data*s)
            input_data = input_data.clamp(max=127)
            input_data = input_data.clamp(min=-128)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        FN, C, FH, FW = self.shape
        N, C, H, W = input_data.shape
        out_h = int(1 + (H + 2*self.padding[0] - FH) / self.stride[0])
        out_w = int(1 + (W + 2*self.padding[1] - FW) / self.stride[1])
        
        col = im2col(input_data, FH, FW, self.stride, self.padding)
        # if self.input_smooth != 1:
        col = col/self.input_smooth
        col_W = self.weight
        # col_W = self.weight.reshape(FN, -1).T

        # col = torch.from_numpy(col).to(device).float()

        # print(col.dtype)
        # print(col_W.dtype)
        out = torch.mm(col, col_W)
        bias = self.bias * self.bias_slice * self.input_slice
        
        out = out + bias

        p = self.output_slice / (self.input_slice * self.weight_slice)
        
        out = torch.round(out*p)
        out = out.clamp(max=127)
        out = out.clamp(min=-128)

        out = out.reshape(N, out_h, out_w, -1).permute(0, 3, 1, 2)
        out = out.to(device)
        return out
    
    # def forward(self, input_data):
    #     if self.input_max_scale != 1:
    #         s = 1/self.input_max_scale
    #         input_data = torch.round(input_data*s)
    #         input_data = input_data.clamp(max=127)
    #         input_data = input_data.clamp(min=-128)

    #     # input_data = input_data*s
    #     x = self.conv(input_data)
        
    #     p = self.output_slice / (self.input_slice * self.weight_slice)
        
    #     x = torch.round(x*p)
    #     x = x.clamp(max=127)
    #     x = x.clamp(min=-128)

    #     return x

        # col = im2col(x, FH, FW, self.stride, self.padding)
        # col_W = self.W.reshape(FN, -1).T.cpu().numpy() # 滤波器的展开
        
        # out = np.dot(col, col_W)
        # out = out + self.b.cpu().numpy()
        # # out = torch.from_numpy(out).to(device)
        # # out = out + self.b
        # # col = torch.from_numpy(col)
        # # col = col.to(device)
        # # print(col.size())
        # # print(col_W.size())
        # # out = torch.mm(col, col_W) + self.b
        
        # out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out = torch.from_numpy(out).to(device)
        # return out

def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的宽
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C, H, W = input_data.shape
    out_h = (H + 2*padding[0] - filter_h)//stride[0] + 1
    out_w = (W + 2*padding[1] - filter_w)//stride[1] + 1

    if input_data.is_cuda: 
        input_data = input_data.cpu()
    img = np.pad(input_data, [(0,0), (0,0), (padding[0], padding[1]), (padding[0], padding[1])], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride[0]*out_h
        for x in range(filter_w):
            x_max = x + stride[1]*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride[0], x:x_max:stride[1]]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    col = torch.from_numpy(col).to(device).float()
    return col



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.float()

        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    
    return model




def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)




def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            module._modules[name] = FusedbnLayer() #Just Identity.
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


def replace_bottlenecks(model, weight_scale, input_scale, output_scale, residual_scale, useSmooth):  
    for name, module in model.named_children():
        # if "Sequential" in str(type(module)):
        if isinstance(module, nn.Sequential):  
            for i, (sub_name, sub_module) in enumerate(module.named_children()):  
                if 'Bottleneck' in str(type(sub_module)):
                    quant_block, scale = quantBottleBlock(name+'.'+sub_name, sub_module, 
                                                          weight_scale, input_scale, output_scale, residual_scale, useSmooth)
                    module._modules[sub_name] = quant_block  



def replace_conv2d_to_Smooth(arch, model, name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, smooth_input):

    for child_name, module in model.named_children():
        if 'conv' in child_name:
                string = name+'.'+child_name
                string = string[1:]
                weight_string = string
                act_scale = 1
                if string == 'conv1':
                    act_scale = input_max_scale
                model._modules[child_name].weight_slice = 1/weight_scale[weight_string]
                model._modules[child_name].input_slice = 1/input_scale[weight_string]
                model._modules[child_name].output_slice = 1/output_scale[weight_string]
                model._modules[child_name].bias_slice = bias_scale[weight_string]/weight_scale[weight_string]
                model._modules[child_name].input_max_scale = act_scale
                # model._modules[child_name].input_smooth = smooth_input[weight_string]
#                 if useconv2d:
#                     scaled_conv2d = QuantConv2d(module, weight_slice=1/weight_scale[weight_string], 
# input_slice=1/input_scale[weight_string], output_slice = 1/output_scale[weight_string], bias_slice = bias_scale[weight_string]/weight_scale[weight_string] , input_max_scale=act_scale)

#                     model._modules[child_name] = scaled_conv2d
#                 else:
#                     model._modules[child_name] = ManualConv2d(module, weight, bias_weight,
#                                                               weight_slice=1/weight_scale[weight_string], input_slice=1/input_scale[weight_string], output_slice = 1/output_scale[weight_string], bias_slice = bias_scale[weight_string]/weight_scale[weight_string], input_max_scale = act_scale)
        else:
            replace_conv2d_to_Smooth(arch, module, name+'.'+child_name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, smooth_input)



def replace_conv2d(arch, model, name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, useconv2d=True):
    for child_name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
                # print("eeeeeeeeee")
                string = name+'.'+child_name
                string = string[1:]
                weight_string = string
                # print(string)
                if module.weight.is_cuda:  
                    weight = module.weight.cpu() 
                else:
                    weight = module.weight
                if module.bias.is_cuda:
                    bias_weight = module.bias.cpu()
                else:
                    bias_weight = module.bias
                weight = weight.detach().numpy()
                bias_weight = bias_weight.detach().numpy()
                act_scale = 1
                if arch == 'resnet50' and string == 'conv1':
                    act_scale = input_max_scale
                elif arch == 'yolov2' and string == 'backbone.conv_1.0.convs.0':
                    act_scale = input_max_scale
                if useconv2d:
                    scaled_conv2d = QuantConv2d(module, weight_slice=1/weight_scale[weight_string], 
                                                input_slice=1/input_scale[weight_string], output_slice = 1/output_scale[weight_string], bias_slice = bias_scale[weight_string]/weight_scale[weight_string] , input_max_scale=act_scale)

                    model._modules[child_name] = scaled_conv2d
                else:
                    model._modules[child_name] = ManualConv2d(module, weight, bias_weight,
                                                              weight_slice=1/weight_scale[weight_string], input_slice=1/input_scale[weight_string], output_slice = 1/output_scale[weight_string], bias_slice = bias_scale[weight_string]/weight_scale[weight_string], input_max_scale = act_scale)
        else:
            replace_conv2d(arch, module, name+'.'+child_name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, useconv2d)


def replace_Avgpool(model, name, weight_scale, input_scale, output_scale):
    for child_name, child in model.named_children():
        if child_name == 'avgpool':
            output_size = child.output_size
            model._modules[child_name] = ManualAdaptiveAvgPool2d(output_size, input_slice=1/input_scale[child_name], output_slice=1/output_scale[child_name])
    return output_scale['avgpool']                                                 #手动设置，对于不同网络可能不同

def replace_Linear(model, weight_scale, input_scale, output_scale, bias_scale, final_scale):
    for child_name, child in model.named_children():
        if child_name == 'fc':
            model._modules[child_name] = ManualLinear(child, weight_slice=1/weight_scale[child_name], input_slice=1/input_scale[child_name], output_slice = 1/output_scale[child_name], bias_slice = bias_scale[child_name]/weight_scale[child_name], final_scale=final_scale)

class ManualAdaptiveAvgPool2d(nn.Module):  
    def __init__(self, output_size=(1, 1), input_slice=1, output_slice=1):  
        super(ManualAdaptiveAvgPool2d, self).__init__()  
        self.output_size = output_size
        self.input_slice = input_slice
        self.output_slice = output_slice
  
    def forward(self, x):  

        batch_size, channels, height, width = x.size()  
            
        kernel_h = height // self.output_size[0]  
        kernel_w = width // self.output_size[1]  
          
        stride_h = kernel_h if height % self.output_size[0] == 0 else kernel_h + 1  
        stride_w = kernel_w if width % self.output_size[1] == 0 else kernel_w + 1  
       
        output_height = (height - kernel_h) // stride_h + 1  
        output_width = (width - kernel_w) // stride_w + 1  
      
        output = torch.zeros(batch_size, channels, output_height, output_width)  
        
        for b in range(batch_size):  
            for c in range(channels):  
                for h in range(0, height-kernel_h+1, stride_h):  
                    for w in range(0, width-kernel_w+1, stride_w):  
                        
                        pool_window = x[b, c, h:h+kernel_h, w:w+kernel_w]
                        pool_window = pool_window * kernel_h * kernel_w                   
                        avg_value = pool_window.mean()
                        output_h = h // stride_h
                        output_w = w // stride_w 
                        output[b, c, output_h, output_w] = avg_value  


        p = self.output_slice / self.input_slice / kernel_h / kernel_w
        output = torch.round(output * p)

        output = output.clamp(max=127)
        output = output.clamp(min=-128)
        return output  





class ManualLinear(nn.Module):  
    def __init__(self, fc_layer, weight_slice=1, input_slice=1, output_slice=1, bias_slice=1,  final_scale=1):  
        super(ManualLinear, self).__init__()
        
        self.fc = nn.Linear(fc_layer.in_features, fc_layer.out_features,
                            bias = True)
        # self.weight = torch.nn.Parameter(torch.randn(output_features, input_features))  
        # self.bias = torch.nn.Parameter(torch.zeros(output_features)) 
        self.weight_slice = weight_slice
        self.input_slice = input_slice
        self.output_slice = output_slice
        self.bias_slice = bias_slice
        self.final_scale = final_scale
        
        self.fc.weight.data.copy_(fc_layer.weight.data)  
        if self.fc.bias is not None:  
            self.fc.bias.data.copy_(fc_layer.bias.data)
            self.fc.bias.data = torch.round(self.fc.bias.data * bias_slice * input_slice)
  
    def forward(self, x):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)

        out = self.fc(x)

        p = self.output_slice / (self.input_slice * self.weight_slice)
        
        x = torch.round(out*p)
        x = x.clamp(max=127)
        x = x.clamp(min=-128)

        scale = 1/self.output_slice
        out = out * scale
        return out



#处理bottleneck和conv的顺序
#先处理bottleneck
def quant_module(arch, model: nn.Module, weight_scale, input_scale, output_scale, residual_scale, input_max_scale, final_scale, bias_scale, smooth_input, useconv2d, useSmooth):
    
    scale = 1
    if arch =='resnet50':
        replace_bottlenecks(model, weight_scale, input_scale, output_scale, residual_scale, useSmooth)
        name = ''
        if useSmooth:
            replace_conv2d_to_Smooth(arch, model, name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, smooth_input)
        else:
            replace_conv2d(arch, model, name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, useconv2d)

        final_scale_2 = 1
        final_scale_2 = replace_Avgpool(model, name, weight_scale, input_scale, output_scale)
        final_scale = final_scale_2 * final_scale
        replace_Linear(model, weight_scale, input_scale, output_scale, bias_scale, final_scale)
    
    elif arch =='yolov2':
        #scales = [output scale layer1, out_scale layer2, out_scale layer3, out_p4, out_p5, out_cat, final_scale/out_pred]
        scales = [1,1,1,1,1,1,1]

        scales[3] = output_scale['reorg']
        scales[4] = output_scale['convsets_1']
        scales[5] = input_scale['convsets_2']
        scales[6] = output_scale['pred']

        model = replace_yolov2model(model,scales)
        name = ''
        if useSmooth:
            replace_conv2d_to_Smooth(arch, model, name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, smooth_input)
        else:
            replace_conv2d(arch, model, name, weight_scale, input_scale, output_scale, input_max_scale, bias_scale, useconv2d)
    
    return model, scale
    

def replace_yolov2model(model, scales):


    from models.yolov2_d19 import QuantYOLOv2D19 as yolo_net
    new_model = yolo_net(device=model.device, 
                   input_size=model.input_size, 
                   num_classes=model.num_classes, 
                   trainable=model.trainable, 
                   anchor_size=model.anchor_size,
                   scales=scales)
    new_model = fuse_conv_bn(new_model)
    # print(new_model)
    # load net
    new_model.load_state_dict(model.state_dict(), strict=False)

    return new_model
    


def quantBottleBlock(name, module, weight_scale, input_scale, output_scale, residual_scale, useSmooth):

    scales = [1,1,1,1]
    if name+'.conv1' in weight_scale:
        scales[0] = output_scale[name+'.conv3']
        if name+'.downsample.0' in weight_scale:
            scales[1] = output_scale[name+'.downsample.0']
        else:
            scales[1] = input_scale[name+'.conv1']
        scales[2] = residual_scale[name]


    downsample = None
    if 'downsample' in module._modules:
        for name, child in module.named_children():
            if name == 'downsample':
                downsample = child
    else:
        downsample = None
    inplanes = 1
    planes = 1
    stride = (1,1)

    for child_name, child in module.named_children():
        if child_name == 'conv1':
            conv1 = child
        elif child_name == 'conv2':
            conv2 = child
        elif child_name == 'conv3':
            conv3 = child

        if hasattr(child, 'stride') and child_name == 'conv2':
            stride = child.stride
        if hasattr(child, 'in_channels') and child_name == 'conv1':
            inplanes = child.in_channels
        if hasattr(child, 'out_channels') and child_name == 'conv1': #conv3 is out_channel * expansion
            planes = child.out_channels

    if useSmooth:
        block = QuantBottleBlock_smooth(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, scales=scales)
        block.conv1 = conv1
        block.conv2 = conv2
        block.conv3 = conv3
        block.load_state_dict(module.state_dict())

    else:
        block = QuantBottleBlock(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, scales=scales)
        block.load_state_dict(module.state_dict())

    return block, 1


class QuantBottleBlock_smooth(nn.Module):
    expansion = 4
    def __init__(self, inplanes=1, planes=1, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, scales=None):
        super(QuantBottleBlock_smooth, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes) * groups
        conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=True)
        self.conv1 = SmoothConv2d(conv1)

        # self.bn1 = norm_layer(width)
        conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)
        self.conv2 = SmoothConv2d(conv2)

        # self.bn2 = norm_layer(width)
        conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=True)
        self.conv3 = SmoothConv2d(conv3)

        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.scales = [1,1,1,1,1,1]
        if scales is not None:
            self.scales = scales

  
    def forward(self, x):  
   
        identity = x
        out = self.conv1(x)

        relu1 = nn.ReLU(inplace=True)
        out = relu1(out)

        out = self.conv2(out)

        relu2 = nn.ReLU(inplace=True)
        out = relu2(out)

        out = self.conv3(out)

        is_downsample = False
        if self.downsample is not None:
            identity = self.downsample(identity)
            is_downsample = True

        out = Add_residual(out, identity, is_downsample, self.scales)
        relu3 = nn.ReLU(inplace=True)
        out = relu3(out)

        return out


# 包含了ResNet原BottleBlock的INT量化版本
class QuantBottleBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes=1, planes=1, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, scales=None):
        super(QuantBottleBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=True)

        # self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)
        
        # self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=True)

        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.scales = [1,1,1,1,1,1]
        if scales is not None:
            self.scales = scales

  
    def forward(self, x):  
   
        identity = x
        out = self.conv1(x)

        relu1 = nn.ReLU(inplace=True)
        out = relu1(out)

        out = self.conv2(out)

        relu2 = nn.ReLU(inplace=True)
        out = relu2(out)

        out = self.conv3(out)

        is_downsample = False
        if self.downsample is not None:
            identity = self.downsample(identity)
            is_downsample = True

        out = Add_residual(out, identity, is_downsample, self.scales)
        relu3 = nn.ReLU(inplace=True)
        out = relu3(out)

        return out


def Add_residual(out, identity, is_downsample, scales):


    residual_slice = 1 / scales[2]
    out = torch.round(out*scales[0] * residual_slice) + torch.round(identity*scales[1]* residual_slice)
    out = out.clamp(max=127)
    out = out.clamp(min=-1)

    return out




class FusedbnLayer(nn.Module):  
    def __init__(self, in_channels = 1, out_channels = 1, scale = 1):  
        super(FusedbnLayer, self).__init__()  
        # self.fc = nn.Linear(in_channels, out_channels)  
        self.scale = scale  
  
    def forward(self, x):  
        x = x
        return x  

class QuantConv2d(nn.Module):
    def __init__(self, conv_layer, weight_slice=1, input_slice=1, output_slice=1, bias_slice=1, input_max_scale=1,final_scale=-1):
        super(QuantConv2d, self).__init__()  
        self.conv = nn.Conv2d(conv_layer.in_channels, conv_layer.out_channels,  
                              conv_layer.kernel_size, conv_layer.stride,  
                              conv_layer.padding, conv_layer.dilation,  
                              conv_layer.groups, conv_layer.bias is not None,  
                              conv_layer.padding_mode)  
        self.input_slice = input_slice
        self.output_slice = output_slice
        self.weight_slice = weight_slice
        self.bias_slice = bias_slice
        self.input_max_scale = input_max_scale
        self.final_scale = final_scale
        self.conv.weight.data.copy_(conv_layer.weight.data)  
        if self.conv.bias is not None:  
            self.conv.bias.data.copy_(conv_layer.bias.data)  
            #self.conv.bias.data = torch.round(self.conv.bias.data * input_slice)
            self.conv.bias.data = torch.round(self.conv.bias.data * bias_slice * input_slice)

    def forward(self, input_data):
        if self.input_max_scale != 1:
            # print(input_data)
            
            # print(self.input_max_scale)
            s = 1/self.input_max_scale
            # print(s)
            input_data = torch.round(input_data*s)
            input_data = input_data.clamp(max=127)
            input_data = input_data.clamp(min=-128)
            # print("nm")
            # print(max(input_data))
        # print()

        # input_data = input_data*s
        x = self.conv(input_data)
        
        p = self.output_slice / (self.input_slice * self.weight_slice)
        
        x = torch.round(x*p)
        # print(torch.max(x), torch.min(x))
        # print(torch.max(x/self.output_slice), torch.min(x/self.output_slice))

        x = x.clamp(max=127)
        x = x.clamp(min=-128)

        # if self.final_scale !=-1:
        #     x = x *self.final_scale
        # print(torch.max(x), torch.min(x))
        # print("??///")
        # assert()
        return x
         
class ManualConv2d(nn.Module):  
    def __init__(self, conv_layer, weight, bias_weight, weight_slice=1, input_slice=1, output_slice=1, bias_slice=1, input_max_scale=1):
        super(ManualConv2d, self).__init__()  
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        if conv_layer.bias is not None:  
            self.bias = bias_weight
        else:  
            self.bias = None
        self.weight = weight
        self.input_slice = input_slice
        self.output_slice = output_slice
        self.weight_slice = weight_slice
        self.bias_slice = bias_slice
        self.input_max_scale = input_max_scale

    def forward(self, input_data):
        s = 1/self.input_max_scale
        input_data = input_data * s
        batch_size, channels, height, width = input_data.shape  
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)


        if input_data.is_cuda: 
            input_data = input_data.cpu()
        padded_input = np.pad(input_data, ((0,), (0,), (self.padding[0],), (self.padding[0],)), 'constant')  
          
        output_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1  
        output_width = (width + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[1] + 1  
        output_data = np.zeros((batch_size, self.out_channels, output_height, output_width))  
          
        for b in range(batch_size):  
            for c in range(self.out_channels):  
                for h in range(0, padded_input.shape[2] - self.kernel_size[0] + 1, self.stride[0]):  
                    for w in range(0, padded_input.shape[3] - self.kernel_size[1] + 1, self.stride[1]):  
                        input_block = padded_input[b, :, h:h+self.kernel_size[0], w:w+self.kernel_size[1]]  
                          
                        # 执行卷积计算  
                        conv_sum = np.sum(input_block * self.weight[c])  
                          
                        # 如果存在偏置项，则加上偏置  
                        if self.bias is not None:  
                            conv_sum +=  round(self.bias[c] * self.input_slice * self.bias_slice)

                        output_data[b, c, h // self.stride[0], w // self.stride[1]] = conv_sum  
        
        p = self.output_slice / (self.input_slice * self.weight_slice)
        output_data = np.round(output_data * p)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output_data = torch.from_numpy(output_data).to(device)

        return output_data  



  

def get_input_scale(model, calib_list):

    device = next(model.parameters()).device
    max_range = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(calib_list):
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # if torch.cuda.is_available():
            #     target = target.cuda(args.gpu, non_blocking=True)
            images = images.to(device)
            max_range = max(max_range, abs(torch.max(images)))
            # compute output
            # output = model(images)
    scale = max_range.item()*2/254
    return scale

def get_weight_scale_and_quant(model):
    # global weight_scale
    weight_scale = {}
    for name, param in model.named_parameters():

        max_value = max(torch.max(param), abs(torch.min(param))).item()
        weight_scale[name] = max_value * 2 / 254
    
    output_weight_scale = {}
    output_bias_scale = {}

    # quant_model = copy.deepcopy(model)
    
    for name, param in model.named_parameters():
        if 'weight' in name:

            layer_name = name[:-len('.weight')]
            scale = weight_scale[name]

            quantized, scale = MSE_update_weight(param, scale)
            output_weight_scale[layer_name] = scale
            param.data = quantized
            # slices = 1 / scale
    
            # clamped_data = torch.round(param * slices)
            # clamped_data = clamped_data.clamp(max=127)
            # clamped_data = clamped_data.clamp(min=-128)
            # param.data = clamped_data


        elif 'bias' in name:
            layer_name = name[:-len('.bias')]
            scale = weight_scale[name]

            quantized, scale = MSE_update_weight(param, scale)
            output_bias_scale[layer_name] = scale
            param.data = quantized

            # output_bias_scale[layer_name] = scale
            # slices = 1 / scale

            # clamped_data = torch.round(param * slices)
            # clamped_data = clamped_data.clamp(max=127)
            # clamped_data = clamped_data.clamp(min=-128)
            # param.data = clamped_data

        # elif 'fc' in name:
        #     param.data = param.data

    
    return model, output_weight_scale, output_bias_scale



def MSE_update_weight(weight, scale):
    best_scaled_weight = weight
    best_scale = scale
    best_mse = -1
    start = 0.5
    stop = 1.1
    step = 0.001
    scale_range = np.arange(start, stop + step, step) 

    for temp in scale_range:
        temp_scale = scale * temp
        quantized_weight = torch.round(weight / temp_scale)
        quantized_weight = quantized_weight.clamp(max=127)
        quantized_weight = quantized_weight.clamp(min=-128)
        quantized_weight = quantized_weight * temp_scale
        mse = torch.mean((weight - quantized_weight) ** 2)
        if best_mse == -1:
            best_mse = mse
            best_scale = temp_scale
        else:
            if mse < best_mse:
                best_mse = mse
                best_scale = temp_scale
    
    # print("Best new scale is {}".format(best_scale/scale))
    best_scaled_weight = torch.round(weight / best_scale)
    best_scaled_weight = best_scaled_weight.clamp(max=127)
    best_scaled_weight = best_scaled_weight.clamp(min=-128)

    return best_scaled_weight, best_scale


def get_scales(arch, model, calib_list):
    model.eval()
    device = next(model.parameters()).device
    input_max = {}
    output_max = {}
    intput_min = {}
    output_min = {}
    residual_max = {}
    
    # input_dict = {}
    # output_dict = {}
    # residual_dict ={}

    def hook_function(module, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
        if not isinstance(input, torch.Tensor) or not isinstance(output, torch.Tensor):  
            return
        # print(name)
        # print(output.shape)
        if arch == 'resnet50':
            max_input_range = max(torch.max(input), abs(torch.min(input))).item()
            max_output_range = max(torch.max(output), abs(torch.min(output))).item()

        elif arch =='yolov2':
            # if isinstance(input, torch.Tensor):  
                max_input_range = torch.max(input).item()
            # if isinstance(output, torch.Tensor):  
                max_output_range = torch.max(output).item()

        if name in input_max:
            input_max[name] = max(input_max[name], max_input_range)
        else:
            input_max[name] = max_input_range
        if name in output_max:
            output_max[name] = max(output_max[name], max_output_range)
        else:
            output_max[name] = max_output_range
        # min_input = torch.min(input).item()
        # min_output = torch.min(output).item()
        # if name in intput_min:
        #     intput_min[name] = min(min_input, intput_min[name])
        # else:
        #     intput_min[name] = min_input
        # if name in output_min:
        #     output_min[name] = min(min_output, output_min[name])
        # else:
        #     output_min[name] = min_output
        
        # if name not in input_dict:
        #     input_dict[name] = []
        # input_dict[name].append(input.cpu())     # many GPU memory used
        # if name not in output_dict:
        #     output_dict[name] = []
        # output_dict[name].append(output.cpu())

    def hook_residual(module, input, output, name):
        max_value = torch.max(output).item()
        if name in residual_max:
            residual_max[name] = max(max_value, residual_max[name])
        else:
            residual_max[name] = max_value
        
        # if name not in residual_dict:
        #     residual_dict[name] = []
        # residual_dict[name].append(output.cpu())
        
    
    hooks = []
    residual_hooks = []
    for name, m in model.named_modules():
        hooks.append(
                m.register_forward_hook(functools.partial(hook_function, name=name))
            )
        if 'Bottleneck' in str(type(m)):
            residual_hooks.append(
                m.register_forward_hook(functools.partial(hook_residual, name=name))
            )
    if arch == 'resnet50':
        with torch.no_grad():
            for i, (images, target) in enumerate(calib_list):
                
                images = images.to(device)
                output = model(images)
    elif arch =='yolov2':
        num_images = len(calib_list)

        for i in range(num_images):
            im, gt, h, w = calib_list.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(device)
            # forward
            bboxes, scores, cls_inds = model(x)


    for h in hooks:
        h.remove()
    for h in residual_hooks:
        h.remove()
    

    input_scale = {}
    output_scale = {}
    residual_scale = {}

    print("Here!")
    print("input!")
    for name, max_range in input_max.items():
        input_scale[name] = max_range*2/254
        print(name)
        print(max_range)
        # input_scale[name] = MSE_update_act(input_dict[name], input_scale[name])
    print("output!")
    for name, max_range in output_max.items():
        output_scale[name] = max_range*2/254
        print(name)
        print(max_range)
        # output_scale[name] = MSE_update_act(output_dict[name], output_scale[name])
    for name, max_range in residual_max.items():
        residual_scale[name] = max_range*2/254
        # residual_scale[name] = MSE_update_act(residual_dict[name], residual_scale[name])
    
    return input_scale, output_scale, residual_scale


def MSE_update_act(origin_calib, scale):
    best_scale = scale
    best_mse = -1
    start = 0.5
    stop = 1.1
    step = 0.001
    scale_range = np.arange(start, stop + step, step) 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for temp in scale_range:
        temp_scale = scale * temp
        mse = 0
        for act in origin_calib:
            act = act.to(device)
            quantized_act = torch.round(act / temp_scale)
            quantized_act = quantized_act.clamp(max=127)
            quantized_act = quantized_act.clamp(min=-128)
            quantized_act = quantized_act * temp_scale
            mse += torch.mean((act - quantized_act) ** 2)

        if best_mse == -1:
            best_mse = mse
            best_scale = temp_scale
        else:
            if mse < best_mse:
                best_mse = mse
                best_scale = temp_scale

    print("Best new scale is {}".format(best_scale/scale))
    
    return  best_scale

# def MSE_update_weight(weight, scale):
#     best_scaled_weight = weight
#     best_scale = scale
#     best_mse = -1
#     start = 1
#     stop = 1
#     step = 0.001
#     scale_range = np.arange(start, stop + step, step) 

#     for temp in scale_range:
#         temp_scale = scale * temp
#         quantized_weight = torch.round(weight / temp_scale)
#         quantized_weight = quantized_weight.clamp(max=127)
#         quantized_weight = quantized_weight.clamp(min=-128)
#         quantized_weight = quantized_weight * temp_scale
#         mse = torch.mean((weight - quantized_weight) ** 2)
#         if best_mse == -1:
#             best_mse = mse
#             best_scale = temp_scale
#         else:
#             if mse < best_mse:
#                 best_mse = mse
#                 best_scale = temp_scale
    
#     # print("Best new scale is {}".format(best_scale/scale))
#     best_scaled_weight = torch.round(weight / best_scale)
#     best_scaled_weight = best_scaled_weight.clamp(max=127)
#     best_scaled_weight = best_scaled_weight.clamp(min=-128)

#     return best_scaled_weight, best_scale





def update_scale(weight_scale, input_scale, output_scale, input_max_scale):

    for key in input_scale:
        if 'conv2' in key:
            last_layer = key[:-len('2')]  + '1'
            input_scale[key] = output_scale[last_layer]
        if 'conv3' in key:
            last_layer = key[:-len('3')]  + '2'
            input_scale[key] = output_scale[last_layer]
    input_scale['layer1.0.conv1'] = output_scale['conv1']

    return input_scale


# The following are unused functions

def test_scale(model, calib_list, weight_scale, input_max_scale):

    model.eval()
    device = next(model.parameters()).device
    input_max = {}
    output_max = {}
    intput_min = {}
    output_min = {}
    
    def hook_function(module, input, output, name):
        if isinstance(input, tuple):
            input = input[0]

        max_input_range = max(torch.max(input), abs(torch.min(input))).item()
        max_output_range = max(torch.max(output), abs(torch.min(output))).item()
        if name in input_max:
            input_max[name] = max(input_max[name], max_input_range)
        else:
            input_max[name] = max_input_range
        if name in output_max:
            output_max[name] = max(output_max[name], max_output_range)
        else:
            output_max[name] = max_output_range
        min_input = torch.min(input).item()
        min_output = torch.min(output).item()
        if name in intput_min:
            intput_min[name] = min(min_input, intput_min[name])
        else:
            intput_min[name] = min_input
        if name in output_min:
            output_min[name] = min(min_output, output_min[name])
        else:
            output_min[name] = min_output
        
    hooks = []

    for name, m in model.named_modules():
        hooks.append(
            m.register_forward_hook(functools.partial(hook_function, name=name))
            )
    print("_________")


    with torch.no_grad():
        for i, (images, target) in enumerate(calib_list):
            images = images.to(device)

            output = model(images)
            break
    for h in hooks:
        h.remove()

        
    print("input_max_scale")
    for key in input_max:
        print("{} input_max: {}, output_max:{}, intput_min:{}, output_min{}".format(key, input_max[key], output_max[key], intput_min[key], output_min[key]))

    input_scale = {}
    output_scale = {}
    
    for name, max_range in input_max.items():
        input_scale[name] = max_range*2/254
    for name, max_range in output_max.items():
        output_scale[name] = max_range*2/254

    return model, input_scale, output_scale

def get_act_scales(model, calib_list):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()

        if name in act_scales:
            print(name)
            print(act_scales[name].shape)
            print(tensor.shape)
            print(comming_max.shape)
            print(act_scales[name])

            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        print(name)
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ReLU): 
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
                )
    with torch.no_grad():
        for i, (images, target) in enumerate(calib_list):
            images = images.to(device)

            output = model(images)
    for h in hooks:
        h.remove()
    return act_scales



def quant_max_pool2d(input_data, kernel_size=3, stride=2, padding=1, output_slices=1):  

    padded_data = np.pad(input_data, ((padding, padding), (padding, padding)), 'constant', constant_values=(0, 0))  
      
    h, w = input_data.shape  
      
    out_h = (h + 2 * padding - kernel_size) // stride + 1  
    out_w = (w + 2 * padding - kernel_size) // stride + 1  
    output_data = np.zeros((out_h, out_w))  
      
    for i in range(0, out_h):  
        for j in range(0, out_w):  
            r_start, r_end = i * stride, i * stride + kernel_size  
            c_start, c_end = j * stride, j * stride + kernel_size  
            region = padded_data[r_start:r_end, c_start:c_end]  
            output_data[i, j] = np.max(region)  
    
    output_data = np.round(output_data * output_slices)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_data = torch.from_numpy(output_data).to(device)

    return output_data


def quant_adaptive_avg_pool2d(input_data, output_slices=1):  

    batch_size, channels, height, width = input_data.shape  
      
    output_data = np.zeros((batch_size, channels, 1, 1))  
    
    for b in range(batch_size):  
        for c in range(channels):  
            sum_value = 0  
            pixel_count = 0  
            for h in range(height):  
                for w in range(width):  
                    sum_value += input_data[b, c, h, w]
                    pixel_count += 1
            
            avg_value = sum_value / pixel_count
            output_data[b, c, 0, 0] = avg_value 


    output_data = np.round(output_data * output_slices)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_data = torch.from_numpy(output_data).to(device)
      
    return output_data  