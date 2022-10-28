from operator import index
import os
import cv2
import gc
import time
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

''' CLIPDataset 알아보기'''
captions_path = "D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/"


dataframe = pd.read_csv("D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/captions.csv")
max_id = dataframe["id"].max() + 1 if not False else 100
image_ids = np.arange(0, max_id)
np.random.seed(42)
valid_ids = np.random.choice(
    image_ids, size=int(0.2 * len(image_ids)), replace=False
)
train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
# print('dataF:', train_dataframe.head())
valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)




captions = list(train_dataframe['caption'])



text_tokenizer = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)


encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=200)

print(encoded_captions)


for key, values in encoded_captions.items() :
    print(key,)
    print('--------------------------')
    print(values)

    
    
'''
df = pd.read_csv("D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/captions.txt")




df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)] # 8091
# print(df['id'].head(20))

df.to_csv("D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/captions.csv", index=False)
df = pd.read_csv("D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/captions.csv")
image_path = "D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/Images"
captions_path = "D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/"




dataframe = pd.read_csv("D:/_AIA_Team_Project_Data/Image_Captioning/_data/Flickr8k/captions.csv")
max_id = dataframe["id"].max() + 1 if not False else 100
image_ids = np.arange(0, max_id)
np.random.seed(42)
valid_ids = np.random.choice(
    image_ids, size=int(0.2 * len(image_ids)), replace=False
)
train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
# print('dataF:', train_dataframe.head())
valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
print(dataframe["image"].values)
'''