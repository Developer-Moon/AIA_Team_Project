from typing import Optional
import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import fastai # 텐서플로나 pytorch 같은 딥러닝 라이브러리
import re
import pickle
from gensim.summarization.summarizer import summarize
import pandas as pd
import joblib as jb

# torch : 1.12.1
# transformers : 4.12.0
# torch text : 0.6.0
# fastai : 2.7.10
# tokenizer : 3.4.1
# typing_extensions : 4.3.0
# gensim : 3.8.3

'''
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


'''
with open('team\KoGPT2\짧은시.txt', encoding='UTF8') as f:
    # lines = f.readlines()
    lines = f.read()

lines = " ".join(lines.split())
lines = re.sub('[\.\"\']', '',lines)
print(len(lines)) # 6655

#split data
train=lines[:int(len(lines)*0.9)]
test=lines[int(len(lines)*0.9):]
splits = [[0],[1]]

#init dataloader
tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
batch, seq_len = 8, 256
dls = tls.dataloaders(bs=batch, seq_len=seq_len)

# compile fit과 비슷함
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp32()
# to_fp16 : mixed-precision training, 부동소수점 방식 loss 사용하겠다는 의미, FP32가 거의 디폴트다
lr=learn.lr_find()
print(lr)
learn.fine_tune(50)

pickle.dump(learn, open(os.path.join('D:\study_data\_data/team_project/korean_written/', 'learn3.pkl'), 'wb'))
# '''

'''
with open(os.path.join('D:\study_data\_data/team_project/korean_written/', 'learn3.pkl'), 'rb') as f:
  learn = pickle.load(f)

prompt= " 두 강아지 "
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()
preds = learn.model.generate(inp,
                           max_length=100,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           repetition_penalty=2.0,       
                           use_cache=True
                          ) 

generated = tokenizer.decode(preds[0].cpu().numpy())
jb.dump(generated, 'team\KoGPT2\generatedtxt.dat')
# '''

generated = jb.load('team\KoGPT2\generatedtxt.dat')
# print(generated)

line = [generated]  # 문장 리스트화
words = line[0].split() # 단어 별로 분리

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
# 각 단어를 다시 한글자씩 쪼개 단어 별 마지막 글자를 검사함
# 종결어미 글자가 일치한다면 그때의 인덱스를 endidx로 저장하고 끝냄

all_words = list(words[:endidx+1])
# endidx까지 단어만 최종 출력에 사용


# 한 행에 NUM_WORD씩 표시하여 시 형태로 출력해주기
sentences=[]
start = 0
NUM_WORD = 4  # 한줄에 출력할 단어 수
for i in range(1, len(all_words)+1):
  if i % NUM_WORD == 0:
    this = all_words[start:i] + ['\n']   # 지정한 글자 수 다음에 엔터 추가
    sentences.append(this)
    start+=NUM_WORD
    end = i
if this != all_words[-1]:
      sentences.append(all_words[end:])  # 엔터 추가한 단어가 마지막 단어가 아니면 남은 단어도 출력하도록 sentences에 append

# print(sentences)
# [['두', '강아지', '뜀뛰는', '가슴의', '\n'], ['너가요', '눈물도', '이', '젊은', '\n'], ['나이를', '눈물로야', '보낼', '거냐', '\n'], 
#  ['나날', '속에서', '오직', '한', '\n'], ['사람의', '이름을', '부르며', '살아온', '\n'], ['그', '누가', '죽어가는가를보다', '풀과', '\n'], 
#  ['나무', '그리고', '산과', '언덕', '\n'], ['온누리', '위에', '던져놓은', '돌이', '\n'], ['끝없이', '연달아']]

# 엔터 추가된 리스트를 모양 잡아서 1문장짜리 리스트로 만들고 4줄씩 연으로 모양 잡기
output = []
for i in range(1, len(sentences)+1):    
  output.append(' '.join(sentences[i-1]))
  if i%4==0: output = output + ['\n'] 
output = ''.join(output)

print(output)

