
## 百度 AI 开放平台 

> 编写：赵君君
  日期：2018-03-12

*基于 Python SDK API*

1.百度语音

- 语音识别 （demo_1）
- 语音合成 （demo_2）
- 语音唤醒

2.视觉技术

- 文字识别 （demo_3）
    - 通用文字识别
    - 通用文字识别（高精度版）
    - 通用文字识别（含位置信息版）
    - 通用文字识别（高精度含位置版）
    - 通用文字识别（含生僻字版）
    - 网络图片文字识别
    - 身份证识别、银行卡识别、驾驶证识别、行驶证识别、车牌识别、营业执照识别、表格文字识别、通用票据识别
    
    
- 人脸识别 （demo_4）
    - 人脸检测 （demo_4_1）
    - 人脸比对 （demo_4_2）
    - 人脸识别(用于计算指定组内用户，与上传图像中人脸的相似度。识别前提为您已经创建了一个人脸库。典型应用场景：如人脸闸机，考勤签到，安防监控等。)

    - 人脸认证
    - ........
- 图像审核
- 图像识别 （demo_5）

    - 菜品识别	检测用户上传的菜品图片，返回具体的菜名、卡路里、置信度信息。
    - 车型识别	检测用户上传的车辆图片，识别所属车型，包括车辆品牌及具体型号。
    - logo商标识别	识别图片中包含的商品LOGO信息，返回LOGO品牌名称、在图片中的位置、置信度。
    - 动物识别	检测用户上传的动物图片，返回动物名称、置信度信息。
    - 植物识别	检测用户上传的植物图片，返回植物名称、置信度信息。
    - 图像主体检测	识别图像中的主体具体坐标位置。
    
- 图像搜索  (....)

3.自然语言

- 语言处理基础技术 （demo_6）
    - 词法分析	分词、词性标注、专名识别
    - 依存句法分析	自动分析文本中的依存句法结构信息
    - 词向量表示	查询词汇的词向量，实现文本的可计算
    - DNN语言模型	判断一句话是否符合语言表达习惯，输出分词结果并给出每个词在句子中的概率值
    - 词义相似度	计算两个给定词语的语义相似度
    - 短文本相似度	判断两个文本的相似度得分
    - 评论观点抽取	提取一个句子观点评论的情感属性
    - 情感倾向分析	对包含主观观点信息的文本进行情感极性类别（积极、消极、中性）的判断，并给出相应的置信度
    - 中文分词	切分出连续文本中的基本词汇序列（已合并到词法分析接口）
    - 词性标注	为自然语言文本中的每个词汇赋予词性（已合并到词法分析接口）

- 理解与交互UNIT
- 文本审核


........

百度 AI 开放平台，提供了十分多的接口，且应用场景，demo 示例，不同语言的 SDK 较为齐全，下面是选取部分接口功能，给出示例 Demo


### 语音识别 （demo_1）


```python
'''

语音识别 （demo_1）

1.ffmpeg 音频文件转码

'''

from aip import AipSpeech
import IPython

""" 你的 APPID AK SK """
APP_ID = '10913130'
API_KEY = '7zKKynArAjNn1a2klRrRzoVP'
SECRET_KEY = 'X4f3W3rRsb2CvgEOVqrqGVCZFGL1gNiX'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 识别本地文件
client.asr(get_file_content('public/16k.pcm'), 'pcm', 16000, {
    'lan': 'zh',
})
```




    {'corpus_no': '6531944375542521720',
     'err_msg': 'success.',
     'err_no': 0,
     'result': ['北京科技馆，'],
     'sn': '300108339431520836813'}



### 语音合成 （demo_2）


```python
'''
语音合成 （demo_2）
'''

result  = client.synthesis('欢迎来到人工智能课程', 'zh', 1, {
    'vol': 5,
})

# 识别正确返回语音二进制 错误则返回dict 参照下面错误码
if not isinstance(result, dict):
    with open('auido.mp3', 'wb') as f:
        f.write(result)

IPython.display.Audio('auido.mp3')    
```





                <audio controls="controls" >
                    <source src="data:audio/mpeg;base64,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" type="audio/mpeg" />
                    Your browser does not support the audio element.
                </audio>
              



### 文字识别 （demo_3）


```python
'''
文字识别 （demo_3）

'''


from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '10913689'
API_KEY = 'OBiEqDl1KpeBGhCv0YkDFwXz'
SECRET_KEY = 'WmquvF1mpWGfrqFqPiAoi3yvYKgnSpz0 '


client1 = AipOcr(APP_ID, API_KEY, SECRET_KEY)


""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('test.jpg')

""" 调用通用文字识别, 图片参数为本地图片 """
client1.basicGeneral(image);

""" 如果有可选参数 """
options = {}
options["language_type"] = "CHN_ENG"
options["detect_direction"] = "true"
options["detect_language"] = "true"
options["probability"] = "true"

""" 带参数调用通用文字识别, 图片参数为本地图片 """
client1.basicGeneral(image, options)



```




    {'direction': 0,
     'language': -1,
     'log_id': 1391583680289477884,
     'words_result': [{'probability': {'average': 0.985094,
        'min': 0.842471,
        'variance': 0.002035},
       'words': '白毛浮绿水,红掌拔清波'}],
     'words_result_num': 1}



[](test.jpg)


```python

url = "http://pic.58pic.com/58pic/11/40/65/42P58PICFW8.jpg"

""" 调用通用文字识别, 图片参数为远程url图片 """
client1.basicGeneralUrl(url);

""" 如果有可选参数 """
options = {}
options["language_type"] = "CHN_ENG"
options["detect_direction"] = "true"
options["detect_language"] = "true"
options["probability"] = "true"

""" 带参数调用通用文字识别, 图片参数为远程url图片 """
client1.basicGeneralUrl(url, options)
```




    {'direction': 0,
     'language': 3,
     'log_id': 7743788044733032176,
     'words_result': [{'probability': {'average': 0.84901,
        'min': 0.738708,
        'variance': 0.012166},
       'words': '零8'},
      {'probability': {'average': 0.998754, 'min': 0.994441, 'variance': 4e-06},
       'words': '(南宋)文天祥'},
      {'probability': {'average': 0.902382, 'min': 0.635468, 'variance': 0.013039},
       'words': '惶恐滩头说惶恐,零丁洋里叹丁'},
      {'probability': {'average': 0.966293, 'min': 0.61964, 'variance': 0.009148},
       'words': '人生自古谁无死,留取丹之照汗青'},
      {'probability': {'average': 0.771932, 'min': 0.771932, 'variance': 0},
       'words': '见'},
      {'probability': {'average': 0.999112, 'min': 0.995501, 'variance': 3e-06},
       'words': '(南宋)陆游'},
      {'probability': {'average': 0.998921, 'min': 0.993217, 'variance': 5e-06},
       'words': '死去元知万事空,'},
      {'probability': {'average': 0.969178, 'min': 0.777893, 'variance': 0.005275},
       'words': '但悲不见九州同。'},
      {'probability': {'average': 0.998663, 'min': 0.993125, 'variance': 6e-06},
       'words': '王师北定中原日'},
      {'probability': {'average': 0.819869, 'min': 0.353516, 'variance': 0.074786},
       'words': '家祭无告乃翁。'}],
     'words_result_num': 10}



### 人脸检测 （demo_4_1）


```python
'''
人脸检测 （demo_4_1）

检测请求图片中的人脸，返回人脸位置、72个关键点坐标、及人脸相关属性信息。

检测响应速度，与图片中人脸数量相关，人脸数量较多时响应时间会有些许延长。

典型应用场景：如人脸属性分析，基于人脸关键点的加工分析，人脸营销活动等。

'''

from aip import AipFace

""" 你的 APPID AK SK """
APP_ID = '10913883'
API_KEY = 'hU9oQMT4v9qWzgyq2BXvVfu5'
SECRET_KEY = 'mS6Bv0DxcTEIsa4odvZ87MDc0vQmMMff'

client2 = AipFace(APP_ID, API_KEY, SECRET_KEY)


""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('face.jpg')

""" 调用人脸检测 """
client2.detect(image);

""" 如果有可选参数 """
options = {}
options["max_face_num"] = 2
options["face_fields"] = "age"

""" 带参数调用人脸检测 """
client2.detect(image, options)


```




    {'log_id': 3368282971031215,
     'result': [{'age': 29,
       'face_probability': 1,
       'location': {'height': 170, 'left': 177, 'top': 139, 'width': 174},
       'pitch': 9.0704860687256,
       'roll': -2.7265648841858,
       'rotation_angle': -2,
       'yaw': 2.0341699123383}],
     'result_num': 1}



### 人脸比对 （demo_4_2）


```python

'''
人脸比对 （demo_4_2） ---安妮·海瑟薇

该请求用于比对多张图片中的人脸相似度并返回两两比对的得分，可用于判断两张脸是否是同一人的可能性大小。

典型应用场景：如人证合一验证，用户认证等，可与您现有的人脸库进行比对验证。

'''

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

images = [
    get_file_content('face.jpg'),
    get_file_content('face1.jpg'),
]

""" 调用人脸比对 """
client2.match(images);

""" 如果有可选参数 """
options = {}
options["ext_fields"] = "qualities"
options["image_liveness"] = ",faceliveness"
options["types"] = "7,13"

""" 带参数调用人脸比对 """
client2.match(images, options)


```




    {'ext_info': {'faceliveness': '0,0.00056620052782819',
      'qualities': '{"occlusion":{"left_eye":0.0096038412302732,"right_eye":0,"nose":0,"mouth":0,"left_cheek":0.048440981656313,"right_cheek":0.021329365670681,"chin":0},"blur":2.7800031721092e-10,"illumination":199,"completeness":1},{"occlusion":{"left_eye":0.0055045871995389,"right_eye":0,"nose":0,"mouth":0,"left_cheek":0.055384613573551,"right_cheek":0.030258519575,"chin":0.0017804154194891},"blur":1.5839526912309e-10,"illumination":158,"completeness":1}'},
     'log_id': 3391578433031215,
     'result': [{'index_i': '0', 'index_j': '1', 'score': 96.851036071777}],
     'result_num': 1}



### 图像识别 （demo_5） --菜品识别


```python
'''
图像识别 （demo_5） --菜品识别

该请求用于菜品识别。即对于输入的一张图片（可正常解码，且长宽比适宜），输出图片的菜品名称、卡路里信息、置信度。

'''

from aip import AipImageClassify

""" 你的 APPID AK SK """
APP_ID = '10914104'
API_KEY = 'SSIafmWWv10qqRblidr2thDq'
SECRET_KEY = 'nDyKq17nxPuOjYEZIKWeSYzx8HgbiHgj'

client3 = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)


""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('kaochuan.jpg')

""" 调用菜品识别 """
client3.dishDetect(image);

""" 如果有可选参数 """
options = {}
options["top_num"] = 3

""" 带参数调用菜品识别 """
client3.dishDetect(image, options)


```




    {'log_id': 1404215582740156209,
     'result': [{'calorie': '206',
       'has_calorie': True,
       'name': '羊肉串',
       'probability': '0.742229'},
      {'calorie': '80',
       'has_calorie': True,
       'name': '骨肉相连',
       'probability': '0.0553208'},
      {'calorie': '0',
       'has_calorie': True,
       'name': '猪肉串',
       'probability': '0.0368781'}],
     'result_num': 3}



### 图像识别 （demo_5） -- 车辆识别


```python
'''
车辆识别

该请求用于检测一张车辆图片的具体车型。即对于输入的一张图片（可正常解码，且长宽比适宜），输出图片的车辆品牌及型号。
'''


""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('car.jpg')

""" 调用车辆识别 """
client3.carDetect(image);

""" 如果有可选参数 """
options = {}
options["top_num"] = 3

""" 带参数调用车辆识别 """
client3.carDetect(image, options)



```




    {'color_result': '蓝色',
     'log_id': 5516450969920487859,
     'result': [{'name': '兰博基尼Aventador',
       'score': 0.99704843759537,
       'year': '2007'},
      {'name': '三菱Endeavor', 'score': 0.0015899568097666, 'year': '2013'},
      {'name': '兰博基尼Murcielago',
       'score': 0.00048711753333919,
       'year': '2004-2010'}]}



### 自然语言 - 语言处理基础技术 （demo_6）


```python
'''
情感倾向分析
对包含主观观点信息的文本进行情感极性类别（积极、消极、中性）的判断，并给出相应的置信度。

'''

from aip import AipNlp

""" 你的 APPID AK SK """
APP_ID = '10914236'
API_KEY = 'CU4ZvBnmXgmFDg0dsCFGdoCd'
SECRET_KEY = 'kCLQIS3u98FFwDfGI5Pxcbz3A23rfiCo'

client4 = AipNlp(APP_ID, API_KEY, SECRET_KEY)

text = "苹果是一家伟大的公司"

""" 调用情感倾向分析 """
client4.sentimentClassify(text)

```




    {'items': [{'confidence': 0.395115,
       'negative_prob': 0.272198,
       'positive_prob': 0.727802,
       'sentiment': 2}],
     'log_id': 2303920266776899402,
     'text': '苹果是一家伟大的公司'}


