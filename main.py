#!/usr/bin/env python
# coding: utf-8

# In[40]:


import random
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F


# In[41]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[42]:


def randomDna():
        "随机的平均的生成DNA序列"
        r=random.random()
        if r<=0.25:
            return 'A'
        elif r>0.25 and r<=0.5:
            return 'G'
        elif r>0.5 and r<=0.75:
            return 'T'
        elif r>0.75 and r<=1:
            return 'C'


# In[43]:


def getdiff(x): #替换
    '处理发生替换错误，返回除x外其他碱基'
    dna=['A','G','T','C']
    if x in dna:
        dna.remove(x)
        return dna[int(random.random()*2)]
    else:
        return 'A'


# In[44]:


#One-hot encoding
def onehotEncoding(strings):
    list1=[]
    for c in strings:
        if c == 'A':
            list1.append([1,0,0,0])
        elif c == 'C':
            list1.append([0,1,0,0])
        elif c == 'G':
            list1.append([0,0,1,0])
        elif c == 'T':
            list1.append([0,0,0,1])
        else:
            print('有错误信息！')
            break;
    return list1
        
        


# In[45]:


#One-hot decoding
def onehotDecoding(x):
    strings=''
    index=x.argmax(dim=1)
    for i in index:
        if i==0:
            strings+='A'
        elif i==1:
            strings+='C'
        elif i==2:
            strings+='G'
        elif i==3:
            strings+='T'
    return strings


# In[46]:


#数据产生函数、产生随机数据并独热编码且相加
def preprocess(dnanum=1000,dnacopenum=100,basenum=200):
    """dnanum母本DNA数、dnacopenum每条DNA副本数量、basenum碱基序列长度 
    返回值为（dnanum,basenum,4）的tenors"""
    #1、随机生成一串 ACGT 序列 
    dna=''
    dnalist=[]
    for i in range(dnanum): #母本DNA数
        for j in range(basenum):#碱基序列长度
            dna=dna+randomDna()
        dnalist.append(dna)
        dna=''
    print("已生成DNA母本")
    #2、生成随机数量的噪声副本和随机数量的错误副本
    
    #生成噪声副本,假设错误率为5%，主要出现的错误是，替换概率为70%，缺失或插入的概率都为15%
    dnacope=''
    dnacopelist=[[]  for i in range(dnanum)]
    for index,dna in enumerate(dnalist): #每条dna都生成副本
        for i in range(dnacopenum): #生成100条错误信息,每条DNA复制100次
            for Base in dna: #循环每个碱基
                if random.random()<=0.05:#如果发生错误
                    errtype=random.random() #判断错误类型
                    if errtype <=0.7 :#发生替换
                        dnacope=dnacope+getdiff(Base)
                    elif errtype > 0.7 and errtype <= 0.85: #插入
                        dnacope=dnacope+randomDna()
                    #缺失 则不添加
                else:#不发生错误
                    dnacope=dnacope+Base
        
            dnacopelist[index].append(dnacope)
            dnacope=''
    print("已生成DNA副本")
    #3数据嵌入
    #过滤过长或过短的副本
    dnacopelist_filtered=[[]  for i in range(dnanum)]
    for index,dnacope in enumerate(dnacopelist): #每条dna的复制序列
        for dna in dnacope:
            if abs(len(dna) - basenum)<=3 : #过滤掉超出指定参数的副本
                dnacopelist_filtered[index].append(dna)
    
    del dnacopelist   #释放空间 
    print("已过滤过长过短的DNA")
    
    #4编码
    #进行独热编码
    dnacopelist_encoding=[[]  for i in range(dnanum)]
    for index,dnacope_filtered in enumerate(dnacopelist_filtered): #每个不同的DNA
        for copeIndex,dna in enumerate(dnacope_filtered):
            dnacopelist_encoding[index].append(onehotEncoding(dna))
            #补足短的副本
            deviation=basenum-len(dna)
            if deviation>0:
                for i in range(deviation): 
                    dnacopelist_encoding[index][copeIndex].append([0,0,0,0])
            #删除过长的副本
            elif deviation<0: 
                for i in range(abs(deviation)):
                    dnacopelist_encoding[index][copeIndex].pop(-1)
    del dnacopelist_filtered  #释放空间
    print("已独热编码")
    #5、副本加和
    sum1=torch.zeros([dnanum,basenum,4],dtype=float)
    for i,dnacope in enumerate(dnacopelist_encoding):
        for dna in dnacope:
            sum1[i]=sum1[i]+torch.tensor(dna)
    print("已加和")
    #6 对目标值独热编码
    dnalist_encoding=[]
    for dna in dnalist:
        dnalist_encoding.append(onehotEncoding(dna))
    label=torch.tensor(dnalist_encoding,dtype=float)
    return {
        'x':sum1,
        'y':label}


# In[47]:


#构建数据集
class DNAData(Dataset):
    def __init__(self,dnanum=3,dnacopenum=100,basenum=200):
        d=preprocess(dnanum,dnacopenum,basenum)
        self.data=d['x']
        self.label=d['y']
    def __len__(self):#返回整个数据集的大小
        return len(self.label)
    def __getitem__(self,index):#根据索引index返回dataset[index]
        return {
            'dna':self.data[index],
            'label':self.label[index]
        }


# In[48]:


#建立模型

class Dnafromer(nn.Module):
    def __init__(self,batch_size,N,M):
        super(Dnafromer, self).__init__()
        self.N=N
        self.batch_size=batch_size #batch_size
        
        #分成四段 进行卷积
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1,padding=0)
        self.conv2 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=5,padding=2)
        self.conv4 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=7,padding=3)
        
        #多层感知机部分 循环3次
        self.liner = nn.Linear(in_features=4,out_features=4)
        self.LN=nn.LayerNorm([100, 4])#归一化
        self.gelu=nn.GELU()#激活函数
        #--liner--
        #--LN--
        #--GELU--
        #--liner--
        #--LN--
        #--GELU--
                              
    
        #--liner--
        
        #transformer
        self.transfromerEncoderLayer=nn.TransformerEncoderLayer(d_model=4,nhead=4)
        self.transformer_encoder=nn.TransformerEncoder(self.transfromerEncoderLayer,num_layers=M)
        #---liner---
        self.softmax=nn.Softmax(dim=2)
        
    def forward(self, x):
        #具体流程 x size:(dnanum,basenum,4)
        x = x.float()
        #拆分 =>(dnanum,4,basenum)
        x2=[[]  for i in range(self.batch_size)]
        for i,each in enumerate(x):
            each=each.t()
        #1、embedding module/convolutions
            x2[i].append(self.conv1(each[0].reshape(1,1,100))[0][0])
            x2[i].append(self.conv1(each[1].reshape(1,1,100))[0][0])
            x2[i].append(self.conv1(each[2].reshape(1,1,100))[0][0])
            x2[i].append(self.conv1(each[3].reshape(1,1,100))[0][0])
            x2[i]=torch.stack(x2[i], 0).t()
        #把张量数组 转为 张量 =>(dnanum,basenum,4)
        x = torch.stack(x2, 0) 
        del x2
        #2、MLP层 N取五
        for i in range(self.N):
            x=self.gelu(self.LN(self.liner(x)))
 
        
        #3、
        x=self.liner(x)
        
        #4、transfromer层 M取五 Encoding
        x=self.transformer_encoder(x)
        
        #5、
        x=self.liner(x)
        x=self.softmax(x)
        return x


# In[49]:


#computing hammingLoss
def HammingLoss(output,label):
    z,y,x=output.size()
    #decoding and encoding
    dnalist=[]
    output_recoding=[]
    
    for eachdna in output: #先把每个碱基中最大评分的设为1，其他设为0
        output_recoding.append(onehotEncoding(onehotDecoding(eachdna)))
   
    output=torch.tensor(output_recoding).to(device)
    del output_recoding
    miss_pairs = sum(sum(sum(output!=label)))
    hammingLoss =torch.tensor(miss_pairs/(x*y*z)).to(device)
    return hammingLoss


# In[50]:


data = DNAData(dnanum=1024,dnacopenum=500,basenum=100)#Dateset实例并模拟生成数据


# In[51]:


batch=16 #原文的batch_size=128
dataloader = DataLoader(data,batch_size=batch,shuffle=True,drop_last=True)#数据加载实例
model=Dnafromer(batch_size=batch,N=2,M=1).to(device) #模型实例
CSEloss= nn.CrossEntropyLoss().to(device)
c1=0.1 #CSEloss 参数
c2=100 #HMloss 参数
optim=torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.9, 0.999)) #原文lr3.141e-5
epoch=10 #训练轮数


# In[52]:


for i in range(epoch):
    print('==========')
    print("第",i+1,"轮")
    for i_batch,batch_data in enumerate(dataloader):
            input=batch_data['dna'].to(device)
            output=model(input)
            y=batch_data['label'].to(device)
            cesloss=CSEloss(output,y)
            hamloss=HammingLoss(output,y)
            res_loss=c1*cesloss+c2*hamloss
            optim.zero_grad()
            res_loss.backward()
            optim.step()
            if (i_batch + 1) % 8 ==0 :
                print(i_batch+1,"batch",'cesloss',cesloss.item(),'hamloss',hamloss.item()
                      ,"res_loss",res_loss.item())


# In[53]:


#测试
#生成测试数据
testdata=DNAData(dnanum=256,dnacopenum=500,basenum=100)
testdataloader = DataLoader(testdata,batch_size=batch,shuffle=True,drop_last=True)


# In[54]:


with torch.no_grad():
    for i,batch_data in enumerate(testdataloader):
        input=batch_data['dna'].to(device)
        output=model(input)
        y=batch_data['label'].to(device)
        cesloss=CSEloss(output,y)
        hamloss=HammingLoss(output,y)
        res_loss=c1*cesloss+c2*hamloss
        print(i+1,"batch",'cesloss',cesloss.item(),'hamloss',hamloss.item(),"res_loss",res_loss.item())


# In[ ]:




