from sklearn.model_selection import train_test_split
import yaml
from glob import glob



with open('C:\home\YoLoV5\datasets\data.yaml') as f:
    _yaml = yaml.load(f, Loader=yaml.FullLoader)
    print(_yaml)

img_list = glob('C:\home\YoLoV5\datasets\export\images\*.jpg')
# print(len(img_list)) # 295

train_img_list, valid_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)
# print(len(train_img_list), len(valid_img_list)) # 236 59

with open('C:\home\YoLoV5\datasets/train.txt', 'w') as f :
    f.write('\n'.join(train_img_list) + '\n')
    
with open('C:\home\YoLoV5\datasets/valid.txt', 'w') as f :
    f.write('\n'.join(valid_img_list) + '\n')    

with open('C:\home\YoLoV5\datasets/data.yaml', 'r') as f :
    data = yaml.safe_load(f) # safe
print(data)
    
data['train'] = 'C:\home\YoLoV5\datasets/train.txt'
data['valid'] = 'C:\home\YoLoV5\datasets/valid.txt'

with open('C:\home\YoLoV5\datasets/data.yaml', 'w') as f :
    yaml.dump(data, f)
print(data)    


    
# python C:\home\YoLoV5/yolov5/train.py --img 416 --batch 16 --epochs 10 --data C:\home\YoLoV5\datasets/data.yaml --cfg C:\home\YoLoV5\yolov5\models\yolov5s.yaml --weights yolov5s.pt --name gun_yolov5s_results
# !python ./yolov5/train.py --img 416 --batch 16 --epochs 10 --data ./dataset/data.yaml --cfg ./yolov5/models/yolov5s.yaml --weights yolov5s.pt --name gun_yolov5s_results