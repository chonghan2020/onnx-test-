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


input_size=640
onnxrun=ONNXModel('yolov7.onnx')
image=cv2.imread('./horses.jpg')
h,w,c=image.shape
ratio = h/w
if ratio > 1 :
    new_h=input_size
    new_w=int(new_h/ratio)
else:
    new_w=input_size
    new_h=int(new_w*ratio)
totensor=tv.transforms.ToTensor()
img=cv2.resize(image,(new_w,new_h))
img=totensor(img.copy())
pad_l = (input_size-new_w)//2
pad_r =input_size-new_w-pad_l
pad_t = (input_size-new_h)//2
pad_d = input_size-new_h-pad_t
img = F.pad(img,(pad_l,pad_r,pad_t,pad_d))
cv_img=(img.permute((1,2,0)).numpy()*255).astype(np.uint8)[:,:,::-1]
cv_img=cv2.UMat(cv_img)
img = img.unsqueeze(0).numpy()

# check input data
# input_img=img.permute((1,2,0)).numpy()
# cv2.imshow("ss",input_img)
# cv2.waitKey(0)


outputs=onnxrun.forward(img)
for i,output in enumerate(outputs):
    outputs[i]=sigmod(output)

strides=[8,16,32]
# yolov7 anchors :
# anchors:
#   - [12,16, 19,36, 40,28]  # P3/8
#   - [36,75, 76,55, 72,146]  # P4/16
#   - [142,110, 192,243, 459,401]  # P5/32
anchors=[[12,16, 19,36, 40,28],[36,75, 76,55, 72,146],[142,110, 192,243, 459,401]]
for i,output in enumerate(outputs):
    stride = strides[i]
    anchor = anchors[i]

    _,f_c,f_h,f_w,nc=output.shape
    for j in range(f_c):
        for k in range(f_h):
            for l in range(f_w):

                confidence = output[0][j][k][l][4]
                output[0][j][k][l][0:2] = (output[0][j][k][l][0:2] * 2 - 0.5 + np.array([l, k])) * stride
                output[0][j][k][l][2:4] = (output[0][j][k][l][2:4] * 2) ** 2 * np.array((anchor[2*j], anchor[2*j+1]))

                if confidence > 0.6:
                    x = output[0][j][k][l][0]
                    y = output[0][j][k][l][1]
                    w = output[0][j][k][l][2]
                    h = output[0][j][k][l][3]
                    cv2.rectangle(cv_img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,255,0),2)

cv2.imshow("ss",cv_img)
cv2.waitKey(0)


print(0)