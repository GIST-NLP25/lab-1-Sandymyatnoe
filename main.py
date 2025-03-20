import pandas as pd
import numpy as np
import csv
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

#####################
# YOU MUST WRITE YOUR STUDENT ID IN THE VARIABLE STUDENT_ID
# EXAMPLE: STUDENT_ID = "12345678"
#####################
STUDENT_ID = "20244096"

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



#checking max length of sentence

def max_length(list_sentences):
    lengths=[]
    for i in list_sentences:
        lengths.append(len(i))
    max_len=max(lengths) 
    return max_len


def unique_dictionary(data_list):
    word_list=[]
    for i in data_list:
        for j in i:
            word_list.append(j)
    unique_word_list=set(word_list)
    dictionary={word:i for i,word in enumerate(unique_word_list)}
    return dictionary



max_length=20
labels=19


def one_hot_encoding(data_list, data_dict):
    one_hot=np.zeros((len(data_list), len(data_dict), max_length))
    for i,sample in enumerate(data_list):
        for j,word in list(enumerate(sample))[:max_length]:
            index=data_dict.get(word)
            one_hot[i, index, j]=1
    return one_hot



def vectorize(onehot_data):
    tensor=torch.from_numpy(onehot_data)
    vectorize=tensor.view(tensor.size(0),-1)
    return vectorize



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
    


def main():
    #train data loading
    file_path_train='./dataset/simple_seq.train.csv'
    train_inputs, train_outputs=load_data(file_path_train)
    check_max_len=max_length(train_inputs)
    train_input_dict=unique_dictionary(train_inputs)

    train_output_dict=unique_dictionary(train_outputs)
    onehot_train_input=one_hot_encoding(train_inputs, train_input_dict)
    onehot_vector_train_input=vectorize(onehot_train_input)
    train_labels=categorize_output(train_outputs, train_output_dict)
    X_train, X_test, y_train, y_test=train_test_split(onehot_vector_train_input, train_labels, test_size=0.2, random_state=42)
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
    


    file_path_test='./dataset/simple_seq.test.csv'
    test_inputs,_=load_data(file_path_test)

    onehot_test_input=one_hot_encoding(test_inputs, train_input_dict)

    onehot_vector_test_input=vectorize(onehot_test_input)


    model.eval()
    with torch.no_grad():
        test_labels=model(onehot_vector_test_input)
        val, pred_test_labels=torch.max(test_labels,1)

    final_df=making_df(pred_test_labels)


    final_df.to_csv('20244096_simple_seq.p1.answer.csv', index=False)


if __name__ == "__main__":
    main()













    
            
        
