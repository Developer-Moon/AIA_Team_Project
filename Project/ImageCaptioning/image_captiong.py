import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.applications import ResNet101, ResNet50

BASE_DIR = 'D:\home_study\Flickr8k_dataset'
WORKING_DIR = './Working'


# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# summarize
model.summary()

'''
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0

 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928

 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856

 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584

 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168

 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080

 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080

 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160

 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808

 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808

 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808

 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808

 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

 flatten (Flatten)           (None, 25088)             0

 fc1 (Dense)                 (None, 4096)              102764544

 fc2 (Dense)                 (None, 4096)              16781312

=================================================================
Total params: 134,260,544
Trainable params: 134,260,544
Non-trainable params: 0
_________________________________________________________________



'''

# 이미지의 features 값을 담아줄 딕셔너리 형태 변수 지정
features = {}
directory = os.path.join(BASE_DIR, 'Images')

# print(os.listdir(directory))

# # 이미지의 features 값을 부여하는 for문 

for img_name in tqdm(os.listdir(directory)): # 해당 이미지 폴더의 이미지리스트에서 하나씩 작업
    # load the image from file
    img_path = directory + '/' + img_name # 각 이미지마다 이미지명으로 경로 지정
    # print(img_path) D:\home_study\Flickr8k_dataset\Images/1000268201_693b08cb0e.jpg
    image = load_img(img_path, target_size=(224, 224)) # 각 이미지를 불러와서 사이즈 지정후 imgae 변수에 담기
    # convert image pixels to numpy array
    image = img_to_array(image) # image 를 numpy 배열로 변환하는 작업
    # print(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # 변환된 값을 reshape 해준다
    
    # preprocess image for vgg
    # print(np.max(image), np.min(image)) # 각 픽셀 채널 범위 0 ~ 255 (원본 이미지 포멧)
    image = preprocess_input(image)
    # print(np.max(image), np.min(image)) # 각 픽셀 채널 범위 -151 ~ 151 (이미지넷 대회에서 사용하는 이미지 포맷)
    # print(image)
    # extract features
    feature = model.predict(image, verbose=1) # image를 vgg16으로 predict
    # print(feature)
    # get image ID
    image_id = img_name.split('.')[0] # 파일명 뒤에 .jpg 확장자 잘라내기
    # store feature
    features[image_id] = feature # 해당 이미지 고유의 predict 값
    
    
# print(features['1000268201_693b08cb0e'])

# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
print('img processing done.')

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
    
    
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f) # 첫줄 빼고 읽어오기 (첫번째 행 무시)
    captions_doc = f.read()

# print(captions_doc)

# create mapping of image to captions

# 이미지와 캡션을 매핑하여 딕셔너리형태로 담기
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')): # enter 기준으로 잘라내서 line으로 담으며 시작
    # split the line by comma(,)
    tokens = line.split(',') # 현재 이미지명.jpg , caption text 가 붙어있어서 , 기준으로 잘라내는것(구분)
    
    if len(line) < 1: # 0 단어짜리 스킵용도. caption.txt 파일 맨마지막에 빈문장자리가 있어서 그거 걸러내는 용도
        continue
    
    # tokens에 담긴 image와 caption을 가져온다
    # print(tokens)
    image_id, caption = tokens[0], tokens[1:] #  0번째만가져옴, 1부터 끝까지 가져옴
    # remove extension from image ID
    # print(image_id, caption) 1000268201_693b08cb0e.jpg ['A child in a pink dress is climbing up a set of stairs in an entry way .']
    image_id = image_id.split('.')[0] # . 이후 지움 (확장자 삭제)
    # convert caption list to string
    # print(caption)
    caption = "".join(caption)
    '''['A rock climber practices on a rock climbing wall .']
            A rock climber practices on a rock climbing wall .'''
    # print(caption)
    
    # create list if needed
    if image_id not in mapping: # key 값으로 mapping 딕셔너리 안에 현재 image_id가 없으면
        mapping[image_id] = []  # image_id:[] 형태로 새로 딕셔너리 자리 하나 만듦
    # store the caption
    mapping[image_id].append(caption) # 만든 자리에 현재 캡션 넣음
                                      # 해당 이미지 id가 이미 있으면 그 자리에 넣음
                                      # 한 이미지당 5개이므로 다음 이미지 아이디가 들어오기 전까지 한자리에 넣음

# 현재 mapping 안에는 이미지 id 와 해당하는 caption이 함께 key, value 로 들어가져있는상태.

# print(len(mapping)) # 8091

# print(mapping['1000268201_693b08cb0e']) 
# print(mapping) 

# '''
# '1000268201_693b08cb0e': ['A child in a pink dress is climbing up a set of stairs in an entry way .', 
# 'A girl going into a wooden building .', 
# 'A little girl climbing into a wooden playhouse .', 
# 'A little girl climbing the stairs to her playhouse .', 
# 'A little girl in a pink dress going into a wooden cabin .'],

# 반복

# '''

# NLP 정제 및 정규화
# 정제, 정규화, 불용어, 소문자변환작업, 불필요한 단어의 제거

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
            
            caption = 'start ' + caption + ' end'
            # 스페이스 기준 잘라서 넣기
            '''a child is standing on her head .
            start a child is standing on her head end .'''
            captions[i] = caption.replace(' .', '') # 마침표 제거
            

# # before preprocess of text
# # print('bf_text:', mapping['1001773457_577c3a7d70'])

# # '''
# # caption 전처리 전 
# # bf_text: ['A child in a pink dress is climbing up a set of stairs in an entry way .', 
# # 'A girl going into a wooden building .', 
# # 'A little girl climbing into a wooden playhouse .', 
# # 'A little girl climbing the stairs to her playhouse .', 
# # 'A little girl in a pink dress going into a wooden cabin .']

# # '''

# # preprocess the text
# clean(mapping)

# # after preprocess of text
# # print('af_text:', mapping['1001773457_577c3a7d70'])

# # '''
# # caption 전처리 후 
# # 특문제거, start end join, 
# # af_text: ['start a black dog and a spotted dog are fighting end', 
# # 'start a black dog and a tri-colored dog playing with each other on the road end', 
# # 'start a black dog and a white dog with brown spots are staring at each other in the street end', 
# # 'start two dogs of different breeds looking at each other on the road end', 
# # 'start two dogs on pavement moving toward each other end']
# # '''

# # # 딕셔너리에서 캡션만 뽑아오기
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
        
# print('all_captions_len:', len(all_captions)) # all_captions_len: 40455

# # print(all_captions[:10]) 

# '''
# ['start a child in a pink dress is climbing up a set of stairs in an entry way end', 
# 'start a girl going into a wooden building end', 
# 'start a little girl climbing into a wooden playhouse end', 
# 'start a little girl climbing the stairs to her playhouse end', 
# 'start a little girl in a pink dress going into a wooden cabin end', 
# 'start a black dog and a spotted dog are fighting end', 
# 'start a black dog and a tri-colored dog playing with each other on the road end', 
# 'start a black dog and a white dog with brown spots are staring at each other in the street end', 
# 'start two dogs of different breeds looking at each other on the road end', 
# 'start two dogs on pavement moving toward each other end']
# '''


# # tokenize the text
# fit_on_text & word_index 를 사용하여 key value로 생성
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1 # 패딩토큰 포함

#fit_on_texts :: 정수인코딩, 단어를 빈도순으로 정렬한뒤 빈도수가 높은 순서대로 차례대로 낮은순서부터 부여하는 방법


# # print(tokenizer.word_index)
# '''
# {'a': 1, 'end': 2, 'start': 3, 'in': 4, 
# 'the': 5, 'on': 6, 'is': 7, 'and': 8, 'dog': 9, 
# 'with': 10, 'man': 11, 
# '''
# print('vacab_size:', vocab_size) # vacab_size: 8494


# # get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
print('max_len:', max_length) # max_len: 38 # max_length 
# padding 을 채우기 위해 제일 긴 길이를 구한다.

# # print(caption.split())


image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90) # train_test_split
train = image_ids[:] # 안함
# test = image_ids[split:]

# # print(len(train)) # 8091

# # # <start> girl going into wooden building end
# # #        X                   y
# # # <start>                   girl
# # # <start> girl              going
# # # <start> girl going        into
# # # ...........
# # # <start> girl going into wooden building      end



# # create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    
    for key in data_keys:
        n += 1
        captions = mapping[key] # key = image_id
        # process each caption
        for caption in captions:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([caption])[0] 
            # 리스트 안에 넣고 (한문장씩 들어가 있으니까)
            # 첫문장을 토크나이징하는 것으로 해야함
            # text 문장을 숫자로 이루어진 리스트로 만든다 
                    
            # split the sequence into X, y pairs
            for i in range(1, len(seq)): # 0 은 start라서
                # split into input and output pairs
                in_seq, out_seq = seq[:i], seq[i] # 현재 문장을 인풋으로, 다음에 올 단어를 아웃풋으로
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0] # 최대 문장 길이만큼 패딩(0을 앞쪽에 채움) maxlen 길이제한.
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0] 
                # 마지막에 소프트맥스값으로 뽑긴 함. 근데 여기서 원핫을 때린다고 원핫밸류가 다르게 찍히는게 이해가 안가는게
                # 여기선 지금 한문장따리만 투카테고리컬에 들어가거든? 그러면 투카테고리컬이 이전 밸류들을 다 기억을 하고 있다는 소린거 같은데 그런가봄
                
                # store the sequences
                X1.append(features[key][0]) # features 에 하나의 key에 해당하는 이미지 피쳐가 리스트로 묶여있기 때문에 인덱스로 부름
                X2.append(in_seq) 
                y.append(out_seq)
        if n == batch_size: # 배치 사이즈만큼 차면 yield로 한묶음 채워서 뱉음
            X1, X2, y = np.array(X1), np.array(X2), np.array(y)
            yield [X1, X2], y
            X1, X2, y = list(), list(), list()
            n = 0

# yield 는 해당 함수가 반복문을 통해 실행 될때마다 차례대로 값을 뱉도록 해준다
# 즉 현재 함수 내에서 while문으로 생성된 yield는 제너레이터형식 주소 안에 차곡차곡 쌓이게 되고
# 함수를 반복해서 부를 때마다 쌓인 yield 가 리턴되는 형태이다
# 그러니까 지금 배치 크기일때 마다 해당 함수의 주소에
# yield1 [X1, X2], y
# yield2 [X1, X2], y
# yield3 [X1, X2], y
# ...
# 이런 형태로 리턴되길 대기하는 중인 것
# while 이 없어도 작동 함

# mapping 에는 이미지 아이디별로 캡션 5개씩 딕셔너리로 되어있고
# features 에는 이미지 아이디별로 VGG16을 통과한 값이 딕셔너리로 되어있음  
  
# encoder model
# image feature layers
inputs1 = Input(shape=(4096,)) # Vgg16 model
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)  

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# train the model
print('start training...')
epochs = 2
batch_size = 32
steps = len(train) // batch_size # 1 batch 당 훈련하는 데이터 수
# len(train): 8091 / steps: 252
# 제너레이터 함수에서 yield로 252개의 [X1, X2], y 묶음이 차곡차곡 쌓여 있고  steps_per_epoch=steps 이 옵션으로
# epoch 1번짜리 fit을 돌때 252번(정해준steps번) generator 를 호출함. iterating 을 steps번 함

for i in range(epochs):
    print(f'epoch: {i+1}')
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1) # generator -> [X1, X2], y
print('done training.')

# save the model
model.save(WORKING_DIR+'/best_model.h5')

def idx_to_word(integer, tokenizer): 
    for word, index in tokenizer.word_index.items(): # word_index :: 단어와 숫자, key values 딕셔너리 반환
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length): # 여기서 image 자리는 vgg 통과해 나온 feature의 자리임
    # add start tag for generation process
    in_text = 'start' # 빈 문장 생성
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0] # 이거 인덱스 없으면 대괄호 하나 더 있어서 4차원이라 LSTM 이 안먹겠다고 오류남
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0) # X1 (feature) / X2 (문장)
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
        if word == 'end':
            break
      
    return in_text

# ''' bleu score
# from nltk.translate.bleu_score import corpus_bleu
# # validate with test data
# actual, predicted = list(), list()

# for key in tqdm(test):
#     # get actual caption
#     captions = mapping[key]
#     # predict the caption for image
#     y_pred = predict_caption(model, features[key], tokenizer, max_length) 
#     # split into words
#     actual_captions = [caption.split() for caption in captions]
#     y_pred = y_pred.split()
#     # append to the list
#     actual.append(actual_captions)
#     predicted.append(y_pred)
    
# # calcuate BLEU score
# print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))        # 1-gram 만 뽑음
# print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))      # 1-gram 과 2-gram 만 뽑되 각각 같은 가중치를 두고 뽑음


# from PIL import Image
# import matplotlib.pyplot as plt
# def generate_caption(image_name):
#     # load the image
#     # image_name = "1001773457_577c3a7d70.jpg"
#     image_id = image_name.split('.')[0]
#     img_path = os.path.join(BASE_DIR, "Images", image_name)
#     image = Image.open(img_path)
#     captions = mapping[image_id]
#     print('---------------------Actual---------------------')
#     for caption in captions:
#         print(caption)
#     # predict the caption
#     y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
#     print('--------------------Predicted--------------------')
#     print(y_pred)
#     plt.imshow(image)
#     plt.show()
# '''

image = load_img('D:\AIA_Team_Project\Project\ImageCaptioning/w1.jpg', target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

print('extracting features..')
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
predic_features = model.predict(image, verbose=1)

print('prediction..')
model = load_model(WORKING_DIR+'/best_model.h5')
y_pred = predict_caption(model, predic_features, tokenizer, max_length)
y_pred = y_pred.replace('start', '')
y_pred = y_pred.replace('end', '')
print(y_pred)

# # generate_caption("1001773457_577c3a7d70.jpg")
# # generate_caption("1002674143_1b742ab4b8.jpg")
# # generate_caption("101669240_b2d3e7f17b.jpg")


# ''' input 시퀀스와 output 시퀀스 쌍
# in: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
#  1]
# out: [0. 0. 0. ... 0. 0. 0.]    # 원핫 상태

# in: [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  3  1 11]
# out: [0. 0. 0. ... 0. 0. 0.]

# in: [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   1
#   11 620]
# out: [0. 0. 0. ... 0. 0. 0.]

# in: [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   1  11
#  620   6]
# out: [0. 1. 0. ... 0. 0. 0.]

# in: [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   1  11 620
#    6   1]
# out: [0. 0. 0. ... 0. 0. 0.]
# '''