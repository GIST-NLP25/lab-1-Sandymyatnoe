#!/usr/bin/env python
# coding: utf-8

# In[377]:


import pandas as pd
import numpy as np
import csv
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


# In[341]:


def load_data(file_path):
    inputs=[]
    outputs=[]
    with open(file_path, 'r') as file:
        reader=csv.reader(file)
        for row in reader:
            line=','.join(row)
            words=line.split(',')
            input_word=[inp for inp in words if inp.startswith('W')]
            output_word=[out for out in words if out.startswith('D')]
            inputs.append(input_word)
            outputs.append(output_word)
    return inputs,outputs


# In[342]:


#train data loading

file_path_train='./test/simple_seq.train.csv'
train_inputs, train_outputs=load_data(file_path_train)


# In[343]:


train_inputs[:5]


# In[344]:


train_outputs


# In[345]:


len(train_inputs)


# In[346]:


#checking max length of sentence

def max_length(list_sentences):
    lengths=[]
    for i in list_sentences:
        lengths.append(len(i))
    max_len=max(lengths) 
    return max_len
    
check_max_len=max_length(train_inputs)


# In[347]:


check_max_len


# In[348]:


def unique_dictionary(data_list):
    word_list=[]
    for i in data_list:
        for j in i:
            word_list.append(j)
    unique_word_list=set(word_list)
    dictionary={word:i for i,word in enumerate(unique_word_list)}
    return dictionary
    


# In[349]:


train_input_dict=unique_dictionary(train_inputs)
train_input_dict


# In[350]:


len(train_input_dict)


# In[351]:


train_output_dict=unique_dictionary(train_outputs)


# In[352]:


train_output_dict


# In[353]:


len(train_output_dict)


# In[354]:


max_length=20
labels=19


# In[355]:


def one_hot_encoding(data_list, data_dict):
    one_hot=np.zeros((len(data_list), len(data_dict), max_length))
    for i,sample in enumerate(data_list):
        for j,word in list(enumerate(sample))[:max_length]:
            index=data_dict.get(word)
            one_hot[i, index, j]=1
    return one_hot


# In[356]:


onehot_train_input=one_hot_encoding(train_inputs, train_input_dict)


# In[357]:


def vectorize(onehot_data):
    tensor=torch.from_numpy(onehot_data)
    vectorize=tensor.view(tensor.size(0),-1)
    return vectorize


# In[358]:


onehot_vector_train_input=vectorize(onehot_train_input)


# In[359]:


#checking train input size

onehot_vector_train_input.size()


# In[369]:


def categorize_output(data_list, data_dict):
    output_list=[]
    categorize=[]
    for i in data_list:
        for j in i:
            output_list.append(j)
    output_cat=[train_output_dict[i] for i in output_list]
    output_tensor=torch.tensor(output_cat)
    output_tensor=output_tensor.to(torch.long)
    return output_tensor
    
            
        
    


# In[370]:


train_labels=categorize_output(train_outputs, train_output_dict)


# In[372]:


train_labels.size()


# In[374]:


X_train, X_test, y_train, y_test=train_test_split(onehot_vector_train_input, train_labels, test_size=0.2, random_state=42)


# In[375]:


len(X_train), len(X_test), len(y_train), len(y_test)


# In[379]:


X_train.size()[1]


# In[383]:


#building 3 layers neural network

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.layer1=nn.Linear(50920, 1000)
        self.layer2=nn.Linear(1000,100)
        self.layer3=nn.Linear(100, labels)
        
        self.batch_norm1=nn.BatchNorm1d(1000)
        self.batch_norm2=nn.BatchNorm1d(100)
        
        init.kaiming_uniform_(self.layer1.weight, a=0, mode='fan_in', nonlinearity='relu')#same with He init in pytorch
        init.kaiming_uniform_(self.layer2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.layer3.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        init.zeros_(self.layer1.bias)
        init.zeros_(self.layer2.bias)
        init.zeros_(self.layer3.bias)
        
    def forward(self,X):
        X=torch.relu(self.batch_norm1(self.layer1(X)))
        X=torch.relu(self.batch_norm2(self.layer2(X)))
        X=self.layer3(X)
        return X


# In[384]:


#training the model

model=NeuralNet().double()
loss_fun=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

num_epochs=200
for epoch in range(num_epochs):
    model.train()
    outputs=model(X_train).double()
    
    loss=loss_fun(outputs,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1)%10==0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    


# In[385]:


#checking accuracy

model.eval()
with torch.no_grad():
    outputs=model(X_test)
    loss=loss_fun(outputs, y_test)
    val, pred_labels=torch.max(outputs,1)
    correct=(pred_labels==y_test).sum().item()
    accuracy=correct/y_test.size(0)
    
    print(f'Test Loss: {loss.item():.4f}')
    print(f'Test Accuracy: {accuracy*100:.4f}%')


# In[386]:


#building test result csv

file_path_test='./test/simple_seq.test.csv'
test_inputs,_=load_data(file_path_test)


# In[387]:


len(test_inputs)


# In[388]:


onehot_test_input=one_hot_encoding(test_inputs, train_input_dict)


# In[391]:


onehot_vector_test_input=vectorize(onehot_test_input)


# In[392]:


onehot_vector_test_input.size()


# In[398]:


model.eval()
with torch.no_grad():
    test_labels=model(onehot_vector_test_input)
    val, pred_test_labels=torch.max(test_labels,1)


# In[404]:


def making_df(test_labels):
    test_labels=test_labels.tolist()
    reverse_dict={value:key for key,value in train_input_dict.items()}
    pred_test_labels=[reverse_dict[i] for i in test_labels]
    ID=[f'S{i:03d}' for i in range(1,101)]
    dataframe=pd.DataFrame({
        'ID':ID,
        'Pred': pred_test_labels
    })
    return dataframe
    


# In[405]:


final_df=making_df(pred_test_labels)


# In[406]:


final_df


# In[407]:


#saving csv file

final_df.to_csv('pred_1.csv', index=False)


# In[ ]:




