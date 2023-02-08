import os
import torch
import json

from PIL import Image
import numpy as np
import skimage

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution

def parse_class(split_file_path):
    with open(split_file_path) as json_file:
        data = json.load(json_file)
    
    class_img_dict = {}
    
    for item in data["train"]:
        if item[2] in class_img_dict:
            class_img_dict[item[2]].append(item[0])
        else:
            class_img_dict[item[2]] = [item[0]]

    for item in data["val"]:
        if item[2] in class_img_dict:
            class_img_dict[item[2]].append(item[0])
        else:
            class_img_dict[item[2]] = [item[0]]

    for item in data["test"]:
        if item[2] in class_img_dict:
            class_img_dict[item[2]].append(item[0])
        else:
            class_img_dict[item[2]] = [item[0]]

    return class_img_dict

def inter_class_visual_variance(class_img_dict):
    class_variance_lst = []
    images = []
    for key in class_img_dict:
        img_lst = class_img_dict[key]
        for img_name in img_lst:
            image = Image.open(os.path.join(data_path, img_name)).convert("RGB")
            images.append(preprocess(image))
        image_input = torch.tensor(np.stack(images)).cuda()
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
        print(image_features.cpu().detach().numpy())
        x_mean = np.mean(image_features, axis = 0)
        print(x_mean)

data_path = "/home/FYP/c190190/DATA/caltech-101/101_ObjectCategories/"
file_path = "/home/FYP/c190190/DATA/caltech-101/split_zhou_Caltech101.json"
class_img_dict = parse_class(file_path)
inter_class_visual_variance(class_img_dict)

    
