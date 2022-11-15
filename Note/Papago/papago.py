import urllib.request
import json

client_id = "EDRKGEUTuKP5ChgXSiVI" # 개발자센터에서 발급받은 Client ID 값
client_secret = "GOFsFnv9W6" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote("We are Teletoby!! hahahaha")
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
    print(result['message']['result']['translatedText'])
else:
    print("Error Code:" + rescode)
    
