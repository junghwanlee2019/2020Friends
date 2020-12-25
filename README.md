# 2020Friends

FRIENDS 감정 데이터 분류

1. 본인의 컴퓨터 혹은 구글 Colab 라이브러리를 DATA PATH로 잡아준다. PATH에 미리 

```
from google.colab import drive 
drive.mount('/content/gdrive') 
DATA_PATH = 'gdrive/My Drive/Colab Notebooks/'

import sys
sys.path.append(DATA_PATH)

!pip install transformers --quiet # package installer for python

import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
pretrained_weights = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertModel.from_pretrained(pretrained_weights)
```



2. 문장을 토큰화해주는 로직을 심는다. BERT에 사용할 수 있도록 문장의 처음과 끝을 잡아준다. 

```
sentence = 'Finally I finished the project.'

tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']
print(tokens)

ids = [tokenizer.convert_tokens_to_ids(tokens)]
print(ids)
input_tensor = torch.tensor(ids)
print(input_tensor.data)
```



3. 위에서 생성된 tensor를 히든 레이어로 만들어준다. 

```
hidden_tensor = model(input_tensor)[0]
print(hidden_tensor.size())

logit = torch.nn.Linear(1024, 2)(hidden_tensor)
print(logit.size())
print(logit.data)

prediction = torch.nn.Softmax(dim=-1)(logit)
print(prediction.size())
print(prediction.data)
```



4. 미리 경로에 넣어둔 json 파일을 불러온다. 

```
import json

data = {'train': {'speaker': [], 'utterance': [], 'emotion': []},
​    'dev': {'speaker': [], 'utterance': [], 'emotion': []},
​    'test': {'speaker': [], 'utterance': [], 'emotion': []}}

for dtype in ['train', 'dev', 'test']:
 for dialog in json.loads(open(DATA_PATH + 'friends_' + dtype + '.json').read()):
  for line in dialog:
   data[dtype]['speaker'].append(line['speaker'])
   data[dtype]['utterance'].append(line['utterance'])
   data[dtype]['emotion'].append(line['emotion'])

test_data = pd.read_csv(DATA_PATH + "en_data.csv", sep=',')

e2i_dict = dict((emo, i) for i, emo in enumerate(set(data['train']['emotion'])))
i2e_dict = {i: e for e, i in e2i_dict.items()}
```

5. 데이터를 학습할 모델을 선언한다

```
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Model(nn.Module):

 def __init__(self):
  super().__init__()
  self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
  self.bert_model = BertModel.from_pretrained(pretrained_weights)
  self.linear = torch.nn.Linear(1024, len(e2i_dict))

 def forward(self, utterance):
  tokens = self.bert_tokenizer.tokenize(utterance)
  tokens = ['[CLS]'] + tokens + ['[SEP]'] # (len)

  ids = [tokenizer.convert_tokens_to_ids(tokens)] # (bat=1, len)
  input_tensor = torch.tensor(ids).cuda()

  hidden_tensor = self.bert_model(input_tensor)[0] # (bat, len, hid)
  hidden_tensor = hidden_tensor[:, 0, :] # (bat, hid)
  logit = self.linear(hidden_tensor)
  
  return logit
```

6. 모델을 평가할 지표를 정의한다. F1 스코어를 최종 지표로 사용할 예정이다. 

```
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(true_list, pred_list):
 precision = precision_score(true_list, pred_list, average=None)
 recall = recall_score(true_list, pred_list, average=None)
 micro_f1 = f1_score(true_list, pred_list, average='micro')
 print('precision:\t', ['%.4f' % v for v in precision])
 print('recall:\t\t', ['%.4f' % v for v in recall])
 print('micro_f1: %.6f' % micro_f1)
```

7. 하이퍼파라메타를 조정해준다. 여기서는 학습률, 에포크수, 버트베이스/라지 를 변수로 한다. 

```import os
pretrained_weights = 'bert-large-uncased'
learning_rate = 1e-6
n_epoch = 1
```

8. 학습을 시작한다

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from tqdm import tqdm_notebook

model = Model()
model.cuda()
criterion = torch.nn.CrossEntropyLoss() # LogSoftmax & NLLLoss
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
for i_epoch in range(n_epoch):
 print('i_epoch:', i_epoch)
 model.train()
 for i_batch in tqdm_notebook(range(len(data['train']['utterance']))):
  logit = model(data['train']['utterance'][i_batch])
  target = torch.tensor([e2i_dict[data['train']['emotion'][i_batch]]]).cuda()
  loss = criterion(logit, target)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

 model.eval()
 pred_list, true_list = [], []
 for i_batch in tqdm_notebook(range(len(data['dev']['utterance']))):
  logit = model(data['dev']['utterance'][i_batch])
  _, max_idx = torch.max(logit, dim=-1)
  pred_list += max_idx.tolist()
  true_list += [e2i_dict[data['dev']['emotion'][i_batch]]]
 evaluate(pred_list, true_list) 
```

9. 최종 결과를 CSV 파일로 내려준다.

```
final_result = []
model.eval()
pred_list, true_list = [], []
for i_batch in tqdm_notebook(range(len(test_data['utterance']))):
 id = test_data['id'][i_batch]
 logit = model(test_data['utterance'][i_batch])
 _, max_idx = torch.max(logit, dim=-1)
 max_idx = int(max_idx.cpu().numpy())

 final_result.append([id , i2e_dict[max_idx]])

final_result[:10]

rdf = pd.DataFrame(final_result, columns =['Id', 'Expected'])
rdf.to_csv(DATA_PATH + 'sample_eng_.csv', index=False)
```

