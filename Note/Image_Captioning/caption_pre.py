from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array # keras로 이지지 불러오기 / 이미지를 넘파이로 변환
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tqdm.notebook import tqdm
import numpy as np
import pickle
import os


BASE_DIR = 'D:\_AIA_Team_Project_Data\_captioning_data\Flickr8k' 
WORKING_DIR = 'D:\_AIA_Team_Project_Data\_captioning_working'    

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # 모델 구조 변경 model.summary()


features = {}
directory = os.path.join(BASE_DIR, 'Images') # BASE_DIR 안의 이미지
'''
for img_name in tqdm(os.listdir(directory)) : 

    
    img_path = directory + '/' + img_name 
    image = load_img(img_path, target_size=(224, 224))                         # load iamge
    image = img_to_array(image)                                                # image -> numpy
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # 4차원으로 shape변환
    
    # preprocess image for vgg
    # print(np.max(image), np.min(image)) # 각 픽셀 채널 범위 0 ~ 255 (원본 이미지 포멧)
    image = preprocess_input(image)
    # print(np.max(image), np.min(image)) # 각 픽셀 채널 범위 -151 ~ 151 (이미지넷 대회에서 사용하는 이미지 포맷)
    
    # extract features
    feature = model.predict(image, verbose=1) # VGG16에 image를 넣은 특징 추출
    # get image ID
    image_id = img_name.split('.')[0]         # image 이름에서 . 기준 0번째 이름만 가져온다 = jpg제거
    # store feature
    features[image_id] = feature              # 현재의 image_id라는 key는 value값으로 지정
    
print(features)


pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
print('img processing done.')
'''
# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f :
     features = pickle.load(f)
    
    
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f : # captions.txt 읽기형식으로 열기
    next(f)                                                   # 두번째 줄 부터 읽기 
    captions_doc = f.read()                                   # 파일 읽기
    

mapping = {}        

for line in tqdm(captions_doc.split('\n')) :   # captions_doc에서 \n 기준으로 split
    # split the line by comma(,)
    tokens = line.split(',')                   # ['997722733_0cb5439472.jpg', 'A rock climber practices on a rock climbing wall .'] , 기준 split (line을 ,로 나눈다)
    if len(line) < 2:                          # 문자 길이가 2미만 일때는 
        continue                               # 하단 for문 무시 후 다음 line로 진행
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]          # . 기준 좌측 문자만 사용
    # convert caption list to string
    caption = " ".join(caption)                # 공백과 caption을 합쳐 caption의 순수 문자만 나오게 한다 (캡션목록을 문자열로 변환)[리스트 상태에서 나오며, ''까지 없어진다]
    # create list if needed
    if image_id not in mapping:                # mapping안에 image_id가 없으면 
        mapping[image_id] = []                 # key=image_id, value는 list로 정의
    # store the caption
    mapping[image_id].append(caption)          # image_id라는 key값에 맞는 caption이라는 value를 더해준다
    

print(len(mapping))


def clean(mapping):
    for key, captions in mapping.items() :             # .items() 함수는 key, value값을 쌍으로 반환
        for i in range(len(captions)) :
            caption = captions[i]                      # captions는 5개의 caption을 가지고 있어서 i번째 caption을 현재 caption으로 지정
            caption = caption.lower()                  # 대문자를 소문자로 변환
            caption = caption.replace('[^A-Za-z]', '') # 정규표현식[^A-Za-z] - ^는 아닌 이라는 뜻 [A-Z] [a-z] : 각각 대문자 알파벳, 소문자 알파벳 모두를 의미, = 소문자, 대문자가 아닌 경우는 공백으로
            caption = caption.replace('\s+', ' ')      # [ \t\n\r\f\v] 가 1번 이상 나오면 공백으로 변경
            caption = caption.replace(' .', '.')
            caption = caption.replace('"', '')
            caption = 'startseq ' + " ".join([word for word in caption.split()]) + ' endseq' # 스페이스 기준 잘라서 넣기
            captions[i] = caption
            

print('bf_text:', mapping['1000268201_693b08cb0e']) # clean 사용 전
clean(mapping)                                      # clean 사용
print('af_text:', mapping['1000268201_693b08cb0e']) # clean 사용 후


all_captions = []                               

for key in mapping:
    for caption in mapping[key] :    # mapping에서 value 개수만큼 for문 
        all_captions.append(caption) # key당 5개의 value값이 나온다, 그 value값들을 all_captions에 더한다
        
# print('all_captions_len:', len(all_captions))

print(all_captions[:10])


# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)       # fit_on_texts() 안에 코퍼스(말뭉치)를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성 print(tokenizer.word_index) 확인 가능
vocab_size = len(tokenizer.word_index) + 1 # 패딩토큰 포함

# print('vacab_size:', vocab_size)         # 8494개

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions) # all_captions 안의 caaption 개수만큼 for문 - 해당 caption의 공백 기준으로 단어를 나눈 단어의 개수들 중 최대값
print('max_len:', max_length) # 38


image_ids = list(mapping.keys())   # mapping의 key값을 list화
split = int(len(image_ids) * 0.90) # list 길이의 90%를 정수화
train = image_ids[:]
# train = image_ids[:split]          # 2780까지 train - 우리는 데이터셋이 적어서 트레인 량을 100프로 쓸꺼다
# test = image_ids[split:]           # 2781부터 test



#                  train
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size) : 
    
    X1, X2, y = list(), list(), list()
    n = 0
    while 1 :
        for key in data_keys :        
            n += 1
            captions = mapping[key]   # key 1개당 5개의 value(caption)값 존재 
            
            for caption in captions : 
                seq = tokenizer.texts_to_sequences([caption])[0] # encode the sequencem, caption을 list 넣지 않으면 알파벳 단위로 토큰화된다
                for i in range(1, len(seq)) :                                      # 토큰화된 seq의 길이만큼 for문
                    in_seq, out_seq = seq[:i], seq[i]                              # in_seq=input, i번째 전 까지의 토큰, out_seq=output, i번째 토큰
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]         # 최대길이를 max_length로 정한 in_seq에 pad로 채우기 
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0] # output 토큰을 vocab_size만큼 one hot 처리 
                    
                    X1.append(features[key][0]) # X1에 feature의 key값에 해당하는 value(특징)을 더해준다 
                    X2.append(in_seq)           
                    y.append(out_seq)
            if n == batch_size : 
                X1, X2, y = np.array(X1), np.array(X2), np.array(y) # 이 list들을 numpy 배열로 변환
                yield [X1, X2], y                                   # yield 저장
                X1, X2, y = list(), list(), list()                  # 하나의 배치가 끝났으므로 다시 초기화
                n = 0
                
                
          
# encoder model - 앙상블 모델
inputs1 = Input(shape=(4096,)) # image feature layers - Vgg16의 model.layers[-2] 
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,)) # sequence feature layers - 
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2) # mask_zero=True를 사용하여 연산시 패딩값으로 채워진 0을 연산에 포함하지 않는다
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3]) 
decoder2 = Dense(256, activation='relu')(decoder1) 
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
# plot_model(model, show_shapes=True) 모델 시각화하기


# train the model

print('start training...')
epochs = 50
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs) :
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1) # steps_per_epoch - 1 epoch당 몇번의 step을 진행할 것, 배치를 몇 번 학습시킬 것이냐
print('done training.')      
                
# save the model
# model.save(WORKING_DIR+'/best_model.h5')

#               yhat,    tokenizer
def idx_to_word(integer, tokenizer) : #integer와 tokenizer
    for word, index in tokenizer.word_index.items() : # all_caption을 토큰화하여 word, index 로 for문 사용 
        if index == integer :
            return word
    return None


# generate caption for an image
#                   model, predic_feature, tokenizer, max_length
def predict_caption(model, image, tokenizer, max_length) :
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length) : 
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

'''
from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
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
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    plt.show()
'''
print('prediction..')
image = load_img('C:\AIA_Team_Project\Project\caption_test/3926232.jpg', target_size=(224, 224))
# image = load_img('D:\_AIA_Team_Project_Data\_captioning_data\Flickr8k\Images/1012212859_01547e3f17.jpg', target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
predic_feature = model.predict(image, verbose=1)

model = load_model(WORKING_DIR+'/best_model.h5')
y_pred = predict_caption(model, predic_feature, tokenizer, max_length)
print(y_pred)

# generate_caption("1001773457_577c3a7d70.jpg")
# generate_caption("1002674143_1b742ab4b8.jpg")
# generate_caption("101669240_b2d3e7f17b.jpg")