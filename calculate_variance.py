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
    
    print("class number is ", len(class_img_dict))

    return class_img_dict

def intra_class_visual_variance(class_img_dict):
    class_variance_lst = []
    variance_sqr_lst = []

    for key in class_img_dict:
        img_lst = class_img_dict[key]
        images = []
        
        for img_name in img_lst:
            image = Image.open(os.path.join(data_path, img_name)).convert("RGB")
            images.append(preprocess(image))

        image_input = torch.tensor(np.stack(images)).cuda()
        # get list of img embs in class c
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        # calculate the mean values of class c
        x_mean = np.mean(image_features, axis = 0)

        # calculate the squared euclidean distance 
        for img_feature in image_features:
            class_variance_lst.append(np.sum(np.square(img_feature - x_mean)))
        
        # calculate the intra class variance of class c
        variance_sqr_lst.append(np.sum(class_variance_lst)/len(class_variance_lst))

    print(len(variance_sqr_lst))
    #calculate the average variance
    intra_class_variance = np.sum(variance_sqr_lst)/len(variance_sqr_lst)
    print("Intra-class Visual Variance is ", intra_class_variance)

def inter_class_text_variance(class_img_dict):
    texts = []
    text_variance_lst = []

    for key in class_img_dict:
        texts.append(key)

    text_tokens = clip.tokenize(texts).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().detach().numpy()
    w_mean = np.mean(text_features, axis = 0)

    for text_feature in text_features:
        text_variance_lst.append(np.sum(np.square(text_feature - w_mean)))

    inter_class_variance = np.sum(text_variance_lst)/len(text_variance_lst)

    print("Inter-class Text Variance is ", inter_class_variance)

# data_path = "/home/FYP/c190190/DATA/caltech-101/101_ObjectCategories/"
# data_path = "/home/FYP/c190190/DATA/stanford_cars/"
# data_path = "/home/FYP/c190190/DATA/oxford_flowers/jpg/"
# data_path = "/home/FYP/c190190/DATA/ucf101/UCF-101-midframes/"
# data_path = "/home/FYP/c190190/DATA/eurosat/2750/"
# data_path = "/home/FYP/c190190/DATA/food-101/images/"
data_path = "/home/FYP/c190190/DATA/oxford_pets/images/"


# file_path = "/home/FYP/c190190/DATA/caltech-101/split_zhou_Caltech101.json"
# file_path = "/home/FYP/c190190/DATA/stanford_cars/split_zhou_StanfordCars.json"
# file_path = "/home/FYP/c190190/DATA/oxford_flowers/split_zhou_OxfordFlowers.json"
# file_path = "/home/FYP/c190190/DATA/ucf101/split_zhou_UCF101.json"
# file_path = "/home/FYP/c190190/DATA/eurosat/split_zhou_EuroSAT.json"
# file_path = "/home/FYP/c190190/DATA/food-101/split_zhou_Food101.json"
file_path = "/home/FYP/c190190/DATA/oxford_pets/split_zhou_OxfordPets.json"

class_img_dict = parse_class(file_path)
intra_class_visual_variance(class_img_dict)
inter_class_text_variance(class_img_dict)
