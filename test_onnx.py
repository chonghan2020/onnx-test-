import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from utils.general import non_max_suppression,xyxy2xywh,scale_coords
from utils.plots import plot_one_box
import argparse

import torchvision as tv


def sigmod(x):
    y=1/(1+np.exp(-x))
    return y


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):

        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        output= self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output

onnxrun=ONNXModel('yolov7.onnx')
image=cv2.imread('./zidane.jpg')
h,w,c=image.shape
ratio = h/w
if ratio > 1 :
    new_h=640
    new_w=int(new_h/ratio)
else:
    new_w=640
    new_h=int(new_w*ratio)
totensor=tv.transforms.ToTensor()
img=cv2.resize(image,(new_w,new_h))
img=totensor(img)
pad_l = (640-new_w)//2
pad_r = 640-new_w-pad_l
pad_t = (640-new_h)//2
pad_d = 640-new_h-pad_t
input_img=F.pad(img,(pad_l,pad_r,pad_t,pad_d))

# check input data
# input_img=input_img.permute((1,2,0)).numpy()

output=onnxrun.forward(input_img)




print(0)