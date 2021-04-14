'''
Description: 
Version: 
Author: Leidi
Date: 2021-03-06 11:18:41
LastEditors: Leidi
LastEditTime: 2021-04-14 13:53:42
'''
from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from Pytorch_Retinaface.data import cfg_mnet, cfg_re50
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from utils.BaseDetector import baseDet
from utils.general import letterbox
import time
import imutils


cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Face_Detect(baseDet):

    def __init__(self):
        super(Face_Detect, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):

        self.weights = 'weights/Resnet50_Final.pth'
        self.model = RetinaFace(cfg=cfg_re50, phase='test')
        self.model = load_model(self.model, self.weights, load_to_cpu=None)
        print('Finished loading model!')
        print(self.model)
        cudnn.benchmark = True
        device = torch.device("cuda")
        self.model = self.model.to(device)

    def preprocess(self, img):

        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(torch.device("cuda"))
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def detect(self, img):

        # 自定义参数
        device = torch.device('cuda')
        cfg = cfg_re50
        resize = 1
        confidence_threshold = 0.8
        top_k = 5000
        nms_threshold = 0.8
        keep_top_k = 750

        start_total = time.time()

        img = np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(torch.device("cuda"))
        scale = scale.to(torch.device("cuda"))

        pre_process_time = time.time()
        print('pre process time: {:.4f}'.format(
            pre_process_time - start_total))
        tic = time.time()
        loc, conf, landms = self.model(img)  # forward pass
        bef_process_time_start = time.time()
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0),
                              prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        for one in dets:
            print(one[4])

        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        # landms = landms[:keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        bef_process_time_end = time.time()
        print('bef process time: {:.4f}'.format(
            bef_process_time_end - bef_process_time_start))
        print('Total time: {:.4f}'.format(bef_process_time_end - start_total))
        print('*'*30)
        return img, dets


def main():

    func_status = {}
    func_status['headpose'] = None
    name = 'demo'

    det = Face_Detect()
    # cap = cv2.VideoCapture(
    #     r'/home/leidi/workspace/detect_sample/running1.mp4')
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    while True:

        _, im = cap.read()
        if im is None:
            break
        im = cv2.resize(im, (640, 480))
        result = det.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)

        cv2.imshow(name, result)
        cv2.waitKey(1)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':

    main()
