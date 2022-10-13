import json
import glob

# 폴더 내 json 파일 이름을 리스트로 저장
# for문을 돌려가며 각 json 파일 내 텍스트 부분을 텍스트파일로 저장
# WBRW1900003252.json 까지 문학

path = 'D:/study_data/_data/team_project/korean_written/NIKL_WRITTEN(v1.1)/'
all_json_names = sorted(glob.glob(path + '*.json'))

# with open('D:/study_data/_data/team_project/korean_written/NIKL_WRITTEN(v1.1)/WBRW1900003155.json', "r", encoding='UTF8') as f:
#     this_json = json.load(f)
    
# print(this_json['document'][0]['paragraph'])
# dic_list = this_json['document'][0]['paragraph']

# all_text = [dic['form'] for dic in dic_list]


all_text = []
for this_file in all_json_names:
    with open(this_file, "r", encoding='UTF8') as f:
        this_json = json.load(f)        
    dic_list = this_json['document'][0]['paragraph']
    this_text = [dic['form'] for dic in dic_list]
    all_text.append(this_text)
    
print(len(all_text[0]))

# 지금 각 파일별 텍스트들이 첫번째파일 텍스트는 리스트 0번째에, 두번째파일 텍스트는 리스트 1번째에 ... 순으로 들어가버려있음
# 텍스트내용을 0번째 리스트에 몰아 넣는 작업이 필요함