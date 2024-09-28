# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
# display plots in this notebook
# %matplotlib inline

import math
import os
import torch
import tqdm
import cv2
import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from .extract_utils import get_image_blob
from bua.caffe import add_bottom_up_attention_config
from bua.caffe.modeling.layers.nms import nms
from bua.d2 import add_attribute_config

MIN_BOXES = 5 # 10
MAX_BOXES = 8 # 20
CONF_THRESH = 0.5 #0.4

data_path = './bottom_up_attention/evaluation'
# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())

config_file = './bottom_up_attention/configs/caffe/test-caffe-r152.yaml'
mode = "caffe"
cfg = get_cfg()
if mode == "caffe":
    add_bottom_up_attention_config(cfg, True)
elif mode == "d2":
    add_attribute_config(cfg)
else:
    raise Exception("detection model not supported: {}".format(mode))
cfg.merge_from_file(config_file)
cfg.merge_from_list(['MODEL.BUA.EXTRACT_FEATS',True])
cfg.freeze()

def load_bottom_model():
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    return model



def model_inference(model, batched_inputs, mode):
    if mode == "caffe":
        return model(batched_inputs)
    elif mode == "d2":
        images = model.preprocess_image(batched_inputs)
        features = model.backbone(images.tensor)
    
        if model.proposal_generator:
            proposals, _ = model.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(model.device) for x in batched_inputs]

        return model.roi_heads(images, features, proposals, None)
    else:
        raise Exception("detection model not supported: {}".format(mode))


def get_boxes(im_file, model):
    im = cv2.imread(im_file)
    dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
    
    model.eval()
    with torch.set_grad_enabled(False):
        boxes, scores, features_pooled, attr_scores = model_inference(model,[dataset_dict],mode)
    # torch.cuda.empty_cache()
        
    dets = boxes[0].tensor.cpu() / dataset_dict['im_scale'] # box_number, 4
    scores = scores[0].cpu() # box_number, 1601
    feats = features_pooled[0].cpu() # box_number, 2048
    attr_scores = attr_scores[0].cpu() # box_number, 401

    max_conf = torch.zeros((scores.shape[0])).to(scores.device) # box_number
    for cls_ind in range(1, scores.shape[1]):
        cls_scores = scores[:, cls_ind]
        keep = nms(dets, cls_scores, 0.15)
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                    cls_scores[keep],
                                    max_conf[keep])
    # keep_boxes 是boxes检测出来全部框的索引
    keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten() # 筛选出置信度大于CONF_THRESH的boxes
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES] # 按照置信度排序
    else:
        keep_boxes = torch.argsort(max_conf, descending=True)[:len(keep_boxes)] # 按照置信度排序
    

    boxes = dets[keep_boxes].numpy() # keep_boxes, 4
    objects = np.argmax(scores[keep_boxes].numpy()[:,1:], axis=1) # object标签，范围是0-1601，keep_boxes
    attr_thresh = 0.1
    attr = np.argmax(attr_scores[keep_boxes].numpy()[:,1:], axis=1) # attributes标签，范围是0-401，keep_boxes
    attr_conf = np.max(attr_scores[keep_boxes].numpy()[:,1:], axis=1) # attributes标签的置信度，范围是0-1，keep_boxes

    center_dis = []
    bbox_es = []
    class_es = []
    for i in range(len(keep_boxes)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        
        x_center = (bbox[0] + bbox[2])/2.0
        y_center = (bbox[1] + bbox[3])/2.0
        dis = math.sqrt((x_center*x_center + y_center*y_center))
        center_dis.append(dis)
        bbox_es.append((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        
        if mode == "caffe":
            cls = classes[objects[i]+1]  # caffe +2
            if attr_conf[i] > attr_thresh: #如果attributes标签的置信度大于0.1，会替换object文本标签为attributes+object文本标签
                cls = attributes[attr[i]+1] + " " + cls   #  caffe +2
        elif mode == "d2":
            cls = classes[objects[i]+2]  # d2 +2
            if attr_conf[i] > attr_thresh:
                cls = attributes[attr[i]+2] + " " + cls   # d2 +2
        else:
            raise Exception("detection model not supported: {}".format(mode))
        class_es.append(cls)
    
    info = {}
    info["bbox"] = bbox_es
    info["class"] = class_es
    info["distance"] = center_dis
    torch.cuda.empty_cache()
        
    return info


# box_dis_index = np.argsort(center_dis, descending=False) # 按照box中心离左上角（0，0）举例排序
