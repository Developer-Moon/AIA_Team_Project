# 가상환경 : project

''' << Version Notice >>
tensorflow : 2.8.2
torch : 1.13.1

for KoGPT2 versions
    - transformers : 4.12.0
    - torch text : 0.6.0
    - fastai : 0.80.0
    - tokenizer : 3.4.1
    - typing_extensions : 4.3.0
'''

import urllib.request
import joblib as jb
import json
import os
import pickle
import numpy as np
import time

from typing import Optional
import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import fastai
import re

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image


####################################################################################################################################
# captioning

BASE_DIR = 'D:\study_data\_data/team_project\Flickr8k/'
WORKING_DIR = 'D:\study_data\_data/team_project\Flickr8k\working/'

'''
# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# summarize
# model.summary()

# extract features from image
features = {}
directory = os.path.join(BASE_DIR, 'Images')

start_time = time.time()
for img_name in os.listdir(directory):
    # load the image from file
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    # preprocess image for vgg
    # print(np.max(image), np.min(image)) 
    image = preprocess_input(image)
    # print(np.max(image), np.min(image)) 
    
    # extract features
    feature = model.predict(image, verbose=1)
    # get image ID
    image_id = img_name.split('.')[0]
    # store feature
    features[image_id] = feature

end_time = time.time() 
print('feature extraction took', end_time-start_time, 'sec.')  
# print(features)


# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
print('img processing done.')
# '''

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
    
    
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()
# print(captions_doc)

# create mapping of image to captions
mapping = {}
# process lines
for line in captions_doc.split('\n'):
    # split the line by comma(,)
    tokens = line.split(',')
    
    if len(line) < 1:
        continue
    
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = "".join(caption)
    '''['A small child is jumping on a bed .\n']
            A small child is jumping on a bed .'''
    
    # create list if needed
    if image_id not in mapping: 
        mapping[image_id] = []  
    # store the caption
    mapping[image_id].append(caption) 

print(len(mapping))

def clean(mapping): # 맵핑 딕셔너리 안의 caption을 전처리
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.
            caption = caption.replace('[^A-Za-z]', '')# [A-Z] [a-z] : 각각 대문자 알파벳, 소문자 알파벳 모두를 의미
            # delete additional spaces
            caption = caption.replace('\s+', ' ') # [ \t\n\r\f\v] 가 1번 이상 나오면 공백으로 변경
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            # 스페이스 기준 잘라서 넣기
            '''a child is standing on her head .
            startseq a child is standing on her head endseq .'''
            captions[i] = caption.replace(' .', '')
            

# before preprocess of text
# print('bf_text:', mapping['1000268201_693b08cb0e'])

# preprocess the text
clean(mapping)

# after preprocess of text
# print('af_text:', mapping['1000268201_693b08cb0e'])


# 딕셔너리에서 캡션만 뽑아오기
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

print('all_captions_len:', len(all_captions))

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
print('vacab_size:', vocab_size) # vacab_size: 8485

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
print('max_len:', max_length) # max_len: 34


image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90) # train_test_split
train = image_ids[:split] # 안함
test = image_ids[split:]

# startseq girl going into wooden building endseq
#        X                   y
# startseq                   girl
# startseq girl              going
# startseq girl going        into
# ...........
# startseq girl going into wooden building      endseq


# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0

    for key in data_keys:
        n += 1
        captions = mapping[key]
        # process each caption
        for caption in captions:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([caption])[0]
                                                             
            # split the sequence into X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pairs
                in_seq, out_seq = seq[:i], seq[i] 
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0] 
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0] 
                
                # store the sequences
                X1.append(features[key][0]) 
                X2.append(in_seq)
                y.append(out_seq)
                
        if n == batch_size: 
            X1, X2, y = np.array(X1), np.array(X2), np.array(y)
            yield [X1, X2], y
            X1, X2, y = list(), list(), list()
            n = 0
'''
# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(128, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = Dense(128)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = LSTM(128)(decoder1)
decoder3 = Dense(32, activation='relu')(decoder2)
outputs = Dense(vocab_size, activation='softmax')(decoder3)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# '''

'''
# train the model
print('start training...')
epochs = 20
batch_size = 40
steps = len(train) // batch_size

start_time = time.time()
for i in range(epochs):
    print(f'epoch: {i+1}')
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1) # generator -> [X1, X2], y
end_time = time.time()
print('done training.')
print('training took', round(end_time-start_time), 'sec.')
print(f'epochs: {epochs}    batch size: {batch_size}')

# save the model
model.save(WORKING_DIR+'/best_model.h5')
# '''

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq' # 빈 문장 생성
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text


image = load_img('D:\study_data\_data/team_project\predict_img/06.jpg', target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

print('extracting features..')
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
predic_features = model.predict(image, verbose=1)

print('prediction..')
# model = load_model(WORKING_DIR+'/best_model.h5')
model = load_model(WORKING_DIR+'/best_model.h5')
y_pred = predict_caption(model, predic_features, tokenizer, max_length)
y_pred = y_pred.replace('startseq', '')
y_pred = y_pred.replace('endseq', '')
print('caption for this img: ', y_pred)

''' bleu score
from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in test:
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length) 
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
    
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))        # 1-gram 만 뽑음
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))      # 1-gram 과 2-gram 만 뽑되 각각 같은 가중치를 두고 뽑음
# '''

###########################################################################################################################

user_caption = y_pred

###########################################################################################################################
# translation

client_id = "EDRKGEUTuKP5ChgXSiVI" # 개발자센터에서 발급받은 Client ID 값
client_secret = "GOFsFnv9W6" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote(user_caption)
data = "source=en&target=ko&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    result = response_body.decode('utf-8')
    result = json.loads(result)
    user_caption_translated = result['message']['result']['translatedText']
    print('before:', user_caption)
    print('translated:', user_caption_translated)
else:
    print("Error Code:" + rescode)
    
###########################################################################################################################
# KoGPT2

# '''
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
model = AutoModelWithLMHead.from_pretrained("skt/kogpt2-base-v2")

#model input output tokenizer
class TransformersTokenizer(Transform):
   def __init__(self, tokenizer): self.tokenizer = tokenizer
   def encodes(self, x): 
       toks = self.tokenizer.tokenize(x)
       return tensor(self.tokenizer.convert_tokens_to_ids(toks))
   def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))
   
#gpt2 ouput is tuple, we need just one val
class DropOutput(Callback):
  def after_pred(self): self.learn.pred = self.pred[0]
# '''


''' << KoGPT2 fine tuning >>
with open('Project\KoGPT2/짧은시.txt', encoding='UTF8') as f:
    # lines = f.readlines()
    lines = f.read()

lines = " ".join(lines.split())
lines = re.sub('[.,\"\']', '',lines)
print(len(lines)) # 6655

#split data
train=lines[:int(len(lines)*0.9)]
test=lines[int(len(lines)*0.9):]
splits = [[0],[1]]

#init dataloader
tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
batch, seq_len = 8, 256
dls = tls.dataloaders(bs=batch, seq_len=seq_len)

learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
lr=learn.lr_find()
print(lr)
learn.fine_tune(50)

pickle.dump(learn, open(os.path.join('D:\study_data\_data/team_project/korean_written/', 'learn2.pkl'), 'wb'))
# '''

with open(os.path.join('D:\study_data\_data/team_project/korean_written/', 'learn2.pkl'), 'rb') as f:
  learn = pickle.load(f)

prompt = re.sub('.', '', user_caption_translated)
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()
preds = learn.model.generate(inp,
                           max_length=128,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           repetition_penalty=2.0,       
                           use_cache=True
                          ) 

generated = tokenizer.decode(preds[0].cpu().numpy())
print(generated)

jb.dump(generated, 'Project\KoGPT2/generatedtxt.dat')
generated = jb.load('Project\KoGPT2/generatedtxt.dat')

line = [generated]
words = line[0].split()

# 한국어 종결어미 맨 마지막 글자 검사 / 최소 LEAST_LEN 단어 이상
LEAST_LEN = 30
for i in range(LEAST_LEN, len(words)):
  if list(words[i])[-1] == '다'\
    or list(words[i])[-1] == '나'\
    or list(words[i])[-1] == '군'\
    or list(words[i])[-1] == '니'\
    or list(words[i])[-1] == '네'\
    or list(words[i])[-1] == '마'\
    or list(words[i])[-1] == '걸'\
    or list(words[i])[-1] == '래'\
    \
    or list(words[i])[-1] == '냐'\
    or list(words[i])[-1] == '련'\
    or list(words[i])[-1] == '랴'\
    or list(words[i])[-1] == '대'\
    or list(words[i])[-1] == '담'\
    \
    or list(words[i])[-1] == '라'\
    or list(words[i])[-1] == '렴'\
    or list(words[i])[-1] == '서'\
    \
    or list(words[i])[-1] == '아'\
    or list(words[i])[-1] == '어'\
    or list(words[i])[-1] == '지'\
    \
    or list(words[i])[-1] == '고'\
    or list(words[i])[-1] == '까'\
    or list(words[i])[-1] == '며':    
      endidx = i
      break

all_words = list(words[:endidx+1])

sentences=[]
start = 0
NUM_WORD = 4  # 한줄에 출력할 단어 수
for i in range(1, len(all_words)+1):
  if i % NUM_WORD == 0:
    this = all_words[start:i] + ['\n']   
    sentences.append(this)
    start+=NUM_WORD
    end = i
if this != all_words[-1]:
      sentences.append(all_words[end:])

txtoutput = []
for i in range(len(sentences)):    
  txtoutput.append(' '.join(sentences[i]))
  if i == 4:
        txtoutput = txtoutput + ['\n'] 
txtoutput = ''.join(txtoutput)

print(txtoutput)
exit()

###########################################################################################################################
# stable diffusion

OUT_DIR = 'C:\AIA_Team_Project\project_program/'
imgfile = OUT_DIR + '/output.png'

def myimshow():
    if os.path.isfile(imgfile):
        img = Image.open(OUT_DIR + '/output.png')
        img.show()
        exit()

myimshow()

STEPS = 50
G_SCALE = 7.5
H, W = 512, 512
SEED = 999

print('generating image..')

generator = StableDiffusion(img_height=H, img_width=W, jit_compile=False)
img = generator.generate(
    user_caption,
    num_steps=STEPS,
    unconditional_guidance_scale=G_SCALE,
    temperature=1,
    batch_size=1,
    seed=SEED,
)
Image.fromarray(img[0]).save(OUT_DIR + '/output.png')
print(f"saved at {OUT_DIR}")

myimshow()
