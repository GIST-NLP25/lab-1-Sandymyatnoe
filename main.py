import pandas as pd
import numpy as np
import csv
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    dev="cuda:0"
else:
    dev='cpu'
    
device=torch.device(dev)
#####################
# YOU MUST WRITE YOUR STUDENT ID IN THE VARIABLE STUDENT_ID
# EXAMPLE: STUDENT_ID = "12345678"
#####################
print("STUDENT_ID =", 20244096)

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        limit = np.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = nn.Parameter(torch.FloatTensor(vocab_size, embed_dim).uniform_(-limit, limit))
    
    def forward(self, indices):
        return self.weight[indices]

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).uniform_(-limit, limit))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        
        return torch.matmul(x, self.weight) + self.bias

class NeuralNet_p1(nn.Module):
    def __init__(self, vocab_size, max_length, num_classes):
        super(NeuralNet_p1, self).__init__()
        
        self.linear1 = LinearLayer(vocab_size*max_length, 2048)  
        self.linear2 = LinearLayer(2048, 512)  
        self.linear3 = LinearLayer(512, num_classes)
        
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)  
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.dropout(self.sigmoid(self.bn1(self.linear1(x))))
        x = self.dropout(self.sigmoid(self.bn2(self.linear2(x))))
        x = self.linear3(x)
        return x


class NeuralNet_p2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(NeuralNet_p2, self).__init__()
        self.embedding = WordEmbedding(vocab_size, embed_dim)
        self.input_dim = embed_dim * 20
        
     
        self.linear1 = LinearLayer(self.input_dim, 2048) 
        self.linear2 = LinearLayer(2048, 512) 
        self.linear3 = LinearLayer(512, num_classes)
        
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.sigmoid(self.bn1(self.linear1(x))))
        x = self.dropout(self.sigmoid(self.bn2(self.linear2(x))))
        x = self.linear3(x)
        return x  

def load_data(file_path):
    inputs, outputs = [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            line = ','.join(row)
            words = line.split(',')
            inputs.append([w for w in words if w.startswith('W')])
            outputs.append([o for o in words if o.startswith('D')])
    return inputs, outputs

def dictionary(data):
    vocab = set()
    for sequence in data:
        vocab.update(sequence)
    dict={word: idx + 1 for idx, word in enumerate(sorted(vocab))}
    return dict

def one_hot_encoding(data_list, data_dict):
    one_hot=np.zeros((len(data_list), len(data_dict)+1,20))
    for i,sample in enumerate(data_list):
        for j,word in list(enumerate(sample)):
            index=data_dict.get(word)
            one_hot[i, index, j]=1
    return one_hot

def vectorize(onehot_data):
    tensor=torch.from_numpy(onehot_data)
    vectorize=tensor.view(tensor.size(0),-1)
    return vectorize

def X_padding(input_sentences,input_dict):
    padding=[[input_dict[w] for w in s] +[0]*(max_length-len(s)) for s in input_sentences]
    return torch.tensor(padding)
    

def categorize_output(data_list):
    output_list = []
    for i in data_list:
        for j in i:
            output_list.append(j)
    
    output_cat = [output_dict[i]-1 for i in output_list]
    output_tensor = torch.tensor(output_cat)
    return output_tensor

def unseen_padding(unseen_data):
    unseen=[[input_dict.get(w,0) for w in s]+[0]*(max_length-len(s)) for s in unseen_data]
    return torch.tensor(unseen)    

def making_df(test_labels):
    idx_to_label = {idx: label for label, idx in output_dict.items()}
    predicted_labels = [idx_to_label[idx.item() + 1] for idx in test_labels] 
    ID=[f'S{i:03d}' for i in range(1,101)]
    dataframe=pd.DataFrame({
        'ID':ID,
        'Pred': predicted_labels
    })
    return dataframe

def train_model(model, X_train, y_train,X_test,y_test):    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_epochs=1500
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss =loss_fn(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()       
        optimizer.step()
        
        if (epoch+1)%100==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        outputs=model(X_test)
        loss = loss_fn(outputs, y_test)
        val, pred_labels=torch.max(outputs,1)
        correct=(pred_labels==y_test).sum().item()
        accuracy=correct/y_test.size(0)
    
        print(f'Test Loss: {loss.item():.4f}')
        print(f'Test Accuracy: {accuracy*100:.4f}%')
max_length = 20
embed_dim = 256  
num_classes = 19
train_inputs, train_outputs = load_data('./dataset/simple_seq.train.csv')
input_dict = dictionary(train_inputs)
output_dict = dictionary(train_outputs)
file_path_test='./dataset/simple_seq.test.csv'
test_inputs,_=load_data(file_path_test)


def main():
    print("Training p1 problem")

    onehot_train_input=one_hot_encoding(train_inputs, input_dict)
    vocab_size=len(onehot_train_input[0])
    onehot_vector_train_input=vectorize(onehot_train_input)
    y=categorize_output(train_outputs)
    y=y.to(torch.long)
    X_train, X_test, y_train, y_test = train_test_split(onehot_vector_train_input, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test=X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
    model1 = NeuralNet_p1(vocab_size,max_length, num_classes).double().to(device)
    train_model(model1, X_train, y_train,X_test,y_test)

    onehot_test_input=one_hot_encoding(test_inputs, input_dict)

    onehot_vector_test_input=vectorize(onehot_test_input).to(device)
    
    model1.eval()
    with torch.no_grad():
        test_labels=model1(onehot_vector_test_input)
        val, pred_test_labels=torch.max(test_labels,1)

    final_df1=making_df(pred_test_labels)
    final_df1.to_csv('20244096_simple_seq.p1.answer.csv', index=False)





    print("Training p2 problem")
    X=X_padding(train_inputs, input_dict)
    y=categorize_output(train_outputs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test=X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
    model2 = NeuralNet_p2(len(input_dict) + 1, embed_dim, num_classes).to(device)
    train_model(model2, X_train, y_train,X_test,y_test)

    X_test_unseen=unseen_padding(test_inputs).to(device)

    


    model2.eval()
    with torch.no_grad():
        test_labels=model2(X_test_unseen)
        val, pred_test_labels=torch.max(test_labels,1)

    final_df2=making_df(pred_test_labels)


    final_df2.to_csv('20244096_simple_seq.p2.answer.csv', index=False)

    




if __name__ == "__main__":
    main()

    













    
            
        
