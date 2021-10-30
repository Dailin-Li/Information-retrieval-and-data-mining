# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:40:21 2021

@author: 20101171
"""
#%%
#import packages
import nltk
import pandas as pd
from pandas import DataFrame
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import copy
import scipy
import random
import xgboost as xgb
#%% load the items we need
#load the data
validation_data=pd.read_csv('validation_data.tsv', sep='\t',header=[0])
train_data=pd.read_csv('train_data.tsv', sep='\t',header=[0])
#load the remove list and stopwords for text pre-processing
remove_list=["'re","'s","'ll","'ve"]
stopword = set(stopwords.words('english')) 
#del stopwords
#%%


PART1



#%%functions had better be loaded at the beginning

#function to calculate average precision
#the input is the ranked relevancy 
def av_precision(relevancy_label,top_N):
    relevancy_label=list(relevancy_label)
    if len(relevancy_label)>=top_N:
        relevancy_label=relevancy_label[0:top_N]
        precision=[] #precision vector
        relevent_num=0 #total relevant label
        for value in range(len(relevancy_label)):
            if relevancy_label[value]==1:
                relevent_num+=1
                precision+=[relevent_num/(value+1)]
        if relevent_num!=0:
            result=sum(precision)/relevent_num
        else:
            result=0       
    else:
        result='NA'

    return(result)

#function to calculate ndcg(normalized discounted cumulative gain)
#input is same with av_precision
def ndcg(relevancy_label,top_N):
    relevancy_label=list(relevancy_label)
    if len(relevancy_label)>=top_N:  
        relevancy_label_ideal=copy.deepcopy(relevancy_label)
        relevancy_label_ideal.sort(reverse=True) #prepare for the idcg
        
        relevancy_label=relevancy_label[0:top_N]
        relevancy_label_ideal=relevancy_label_ideal[0:top_N]
        dcg=[]
        idcg=[]
        for value in range(len(relevancy_label)):
            dcg+=[relevancy_label[value]/np.log2(value+2)]
            idcg+=[relevancy_label_ideal[value]/np.log2(value+2)]
        result= sum(dcg)/sum(idcg)
    else:
        result='NA'

    return(result)


def plot_acc(df,list_name,method_name,title_name):

    df=df.fillna('NA')
    colour=["red","blue","green","black","purple","orange","yellow","white"]
    for i,name in enumerate(list_name):
        list_i=list(df[name][(df[name]!='NA')])
        list_i.sort(reverse=True)
        plt.plot(list_i,color=colour[i], label=name)
    
    plt.title(title_name)
    plt.legend()
    plt.xlabel('Query')
    plt.ylabel(method_name)
    plt.show()
    
#%%load the data set(could replace the given dataset)
#And this dataset could be either loaded at the beginning or trained in part1 
df_train_data=pd.read_csv('df_train_data.tsv', sep='\t',header=[0])
df_validation_data=pd.read_csv('df_validation_data.tsv', sep='\t',header=[0])
df_train_test=pd.read_csv('df_train_test.tsv', sep='\t',header=[0])
df_validation_test=pd.read_csv('df_validation_test.tsv', sep='\t',header=[0])
#train data (balanced version)
df_train_data=pd.read_csv('train_sample.tsv', sep='\t',header=[0]) #load
#%%Impport BM25 function (copy from ICA1)
def BM25_score(text_query,dict_query_doc,dict_query_inv,k1=1.2,k2=5,b=0.75):
    
    #calculate N (number of documents)
    N=len(dict_query_doc)
    #set a dictionary to store the term weight in BM25 and name it as BM25_part1
    dict_BM25_part1 = {}   
    for word,dict in dict_query_inv.items():
        dict_BM25_part1[word] = np.log((N-len(dict)+0.5)/(len(dict)+0.5))
        
    #set a dictionary to calculate dl (document length)
    dict_dl={}
    for pid,dict in dict_query_doc.items():
        dict_dl[pid] = sum(dict.values())
    #calculate avdl (average document length)
    avdl=sum(dict_dl.values())/len(dict_dl)
    
    #calculate the count of each query term and store it in dict
    dict_qf=Counter(text_query)
    
    #set dictionary to store the BM25 score of each document
    dict_score={}
    for pid , dict in  dict_query_doc.items():
        score=0
        #calculate parameter K (depending on k1,b,dl and avdl) for each document
        K=k1*((1-b)+b*dict_dl[pid]/avdl)
        #start for-loop of query text        
        for word in list(set(text_query)):
            #calculate f (counts of word in the doc) and find its weight in dict
            if word in list(dict.keys()): #if the word in the doc
                BM25_part1=dict_BM25_part1[word]
                f=dict[word]
            else:
                f=0
                if word in list(dict_query_inv.keys()):#if the word in the collect
                    BM25_part1=dict_BM25_part1[word]
                else:
                    BM25_part1=np.log((N+0.5)/0.5)
            #take the counts of query term in query             
            qf=dict_qf[word]
            #calculate the score of BM25
            score+=BM25_part1*((k1+1)*f/(K+f))*((k2+1)*qf/(k2+qf))
        #store the score
        dict_score[pid]=score    
    return dict_score


#define a function to process the text of query to maintain the same form with our documents
def query_proc(query,stopword,remove_list):
    output_query=[]
    for word in word_tokenize(query):
        word_low = word.lower()
        #abandon the complete symbol term&stopwords
        if (word_low not in stopword) and (bool(re.search(r'[a-z]',word_low))):
            #stemming
            word_stem = PorterStemmer().stem(word_low)
            #remove term in the remove list
            if word_stem not in remove_list:
                #add the word to output_query
                output_query+=[word_stem]
    return output_query

def dict_con(df,col_name_passage,col_name_pid,stopword,remove_list):
    #term is stored in key of dict, value is its occurance(pid).
    dict_term_invert=defaultdict(list) 
    #term is stored in value of dict, key is the position of document (pid)
    dict_doc=defaultdict(list)#dictionary for documents
    
    #set the pid list
    pid=list(df[col_name_pid])
    
    for i, doc in enumerate(df[col_name_passage]):
        for word in word_tokenize(doc):
            word_low = word.lower()
            #abandon the complete symbol/num term&stopwords
            if (word_low not in stopword) and (bool(re.search(r'[a-z]',word_low))):
                #stemming
                word_stem = PorterStemmer().stem(word_low)
                #remove term in the remove list
                if word_stem not in remove_list:
                    #add the inverted index to the dictionary
                    dict_term_invert[word_stem].append(pid[i])
                    dict_doc[pid[i]].append(word_stem)
    #change the repeated positions into position together with counts
    for i in dict_term_invert:
        dict_term_invert[i]=Counter(dict_term_invert[i])
    #change the repeated terms into term together with counts
    for i in dict_doc:
        dict_doc[i]=Counter(dict_doc[i])
    return dict_term_invert,dict_doc

def process_dict_score(dict_score,df_query):   
    #change the form of score from dict to dataframe
    df_score= pd.DataFrame.from_dict(dict_score, orient='index')
    df_score.columns=['score']
    #join the score dataframe with 'top1000' (under certain qid) by passage id 
    df_query_score=pd.merge(df_query,df_score,how='inner', left_on='pid', right_index=True)  
    #rank the dataframe by the score
    df_query_score=df_query_score.sort_values(by=['score'], ascending=False)
    #add the assignment column
    df_query_score['assignment']='A2'
    #decide the final output
    df_query_score=df_query_score.loc[:,['qid','assignment','pid','queries','passage','score','relevancy']]
    
    return(df_query_score)


#%%calculate BM25 and evaluate the result of validation_data

#summary the unqiue query id (Similar to "test" in ICA1)
df_validation_test=validation_data.loc[:,['qid','queries']]
df_validation_test.drop_duplicates(subset=None, keep='first', inplace=True)

#preserve av_precision/ndcg in the df_validation_test
df_validation_test['BM25_av_precision']='NA'
df_validation_test['BM25_ndcg']='NA'

#add a BM25 score column to "validation_data" and name it as df_validation_data
df_validation_data=pd.DataFrame()


queries_loc=df_validation_test.columns.get_loc('queries')
for i,qid in enumerate(df_validation_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_validation_test.index)))   
    df_query=validation_data[validation_data['qid']==qid]  
    #take the text of query
    text_query= df_validation_test.iloc[i,queries_loc]
    #process the text to maintain the same form with our documents
    text_query=query_proc(text_query,stopword,remove_list)
    
    #construct the term dictionary and the doc dictionary for this qid
    dict_query_inv,dict_query_doc=dict_con(df_query,'passage','pid',stopword,remove_list)
    
    #calculate the score using this method under different passages (docs)
    dict_score=BM25_score(text_query,dict_query_doc,dict_query_inv,k1=1.2,k2=5,b=0.75)
    #process the score of dictionary of this method
    df_query_score=process_dict_score(dict_score,df_query) 
    
    #add the df_query_score to the whole df_validation_data
    df_validation_data = pd.concat([df_validation_data,df_query_score])
    
    #calculate average precision/ndcg for each query
    df_validation_test.loc[df_validation_test['qid']==qid,'BM25_ndcg']=ndcg(df_query_score['relevancy'],100)
    df_validation_test.loc[df_validation_test['qid']==qid,'BM25_av_precision']=av_precision(df_query_score['relevancy'],100)

df_validation_data=df_validation_data.rename(columns={'score':'BM25_sc'})



#%%calculate BM25 and evaluate the result of train_data

#summary the unqiue query id (Similar to "test" in ICA1)
df_train_test=train_data.loc[:,['qid','queries']] 
df_train_test.drop_duplicates(subset=None, keep='first', inplace=True)

#preserve av_precision/ndcg in the df_train_test
df_train_test['BM25_av_precision']='NA'
df_train_test['BM25_ndcg']='NA'

#add a BM25 score column to "train_data" and name it as df_train_data
df_train_data=pd.DataFrame()


queries_loc=df_train_test.columns.get_loc('queries')
for i,qid in enumerate(df_train_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_train_test.index)))   
    df_query=train_data[train_data['qid']==qid]
    #take the text of query
    text_query=df_train_test.iloc[i,queries_loc]
    #process the text to maintain the same form with our documents
    text_query=query_proc(text_query,stopword,remove_list)
    
    #construct the term dictionary and the doc dictionary for this qid
    dict_query_inv,dict_query_doc=dict_con(df_query,'passage','pid',stopword,remove_list)
    
    #calculate the score using this method under different passages (docs)
    dict_score=BM25_score(text_query,dict_query_doc,dict_query_inv,k1=1.2,k2=5,b=0.75)
    #process the score of dictionary of this method
    df_query_score=process_dict_score(dict_score,df_query) 
    
    #add the df_query_score to the whole df_train_data
    df_train_data = pd.concat([df_train_data,df_query_score])
    
    #calculate average precision/ndcg for each query
    df_train_test.loc[df_train_test['qid']==qid,'BM25_ndcg']=ndcg(df_query_score['relevancy'],100)
    df_train_test.loc[df_train_test['qid']==qid,'BM25_av_precision']=av_precision(df_query_score['relevancy'],100)

df_train_data=df_train_data.rename(columns={'score':'BM25_sc'})



#%%store the data set 

df_train_data.to_csv('df_train_data.tsv',sep='\t',index=False,header=True)
df_validation_data.to_csv('df_validation_data.tsv',sep='\t',index=False,header=True)
df_train_test.to_csv('df_train_test.tsv',sep='\t',index=False,header=True)
df_validation_test.to_csv('df_validation_test.tsv',sep='\t',index=False,header=True)

#%% simulate the random rank result
df_validation_test['Random_ndcg']='NA'
df_validation_test['Random_av_precision']='NA'
for i,qid in enumerate(df_validation_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_validation_test.index)))   
    df_query=df_validation_data[df_validation_data['qid']==qid]
    df_query=df_query.iloc[random.sample(list(range(len(df_query))),len(df_query)),:]
    
    #calculate average precision/ndcg for each query
    df_validation_test.loc[df_validation_test['qid']==qid,'Random_ndcg']=ndcg(df_query['relevancy'],100)
    df_validation_test.loc[df_validation_test['qid']==qid,'Random_av_precision']=av_precision(df_query['relevancy'],100)

df_train_test['Random_ndcg']='NA'
df_train_test['Random_av_precision']='NA'
for i,qid in enumerate(df_train_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_train_test.index)))   
    df_query=df_train_data[df_train_data['qid']==qid]
    df_query=df_query.iloc[random.sample(list(range(len(df_query))),len(df_query)),:]
    
    #calculate average precision/ndcg for each query
    df_train_test.loc[df_train_test['qid']==qid,'Random_ndcg']=ndcg(df_query['relevancy'],100)
    df_train_test.loc[df_train_test['qid']==qid,'Random_av_precision']=av_precision(df_query['relevancy'],100)

#plot the comparison between BM25 and random-rank in different dataset and different method
plot_acc(df_validation_test,['BM25_ndcg','Random_ndcg'],'NDCG Value','Plot to assess accuracy (Validaition)')
plot_acc(df_validation_test,['BM25_av_precision','Random_av_precision'],'Average Precision Value','Plot to assess accuracy (Validaition)')

plot_acc(df_train_test,['BM25_ndcg','Random_ndcg'],'NDCG Value','Plot to assess accuracy (Train)')
plot_acc(df_train_test,['BM25_av_precision','Random_av_precision'],'Average Precision Value','Plot to assess accuracy (Train)')

#%%


PART2



#%%Select a smaller sample if needed

df_train_test=df_validation_test.iloc[201:,:]
df_validation_test=df_validation_test.iloc[0:201,:]

df1=pd.DataFrame()
for i,qid in enumerate(df_train_test['qid']):
    df1=pd.concat([df1,df_validation_data[df_validation_data['qid'] == qid]])
df2=pd.DataFrame()
for i,qid in enumerate(df_validation_test['qid']):
    df2=pd.concat([df2,df_validation_data[df_validation_data['qid'] == qid]])
    
df_train_data=df1
df_validation_data=df2

df_train_data.to_csv('train.tsv',sep='\t',index=False,header=True)
df_validation_data.to_csv('test.tsv',sep='\t',index=False,header=True)
#%%balance the sample if needed

#set the number of 0 is x times of the number of 1
def make_data_balance(df, x=3):
    df_output=pd.DataFrame()
    qid_list=list(set(df['qid']))
    for i,qid in enumerate(qid_list):
        
        print("Processing progress is %s '%%'" %(100*i/len(list(set(df['qid']))))) 
        df_query=df[df['qid']==qid]
        #position of 1    
        position1=list(df_query[df_query['relevancy']==1].index)
        num=x*len(position1)
        
        if num<(df_query.shape[0]-len(position1)):
            #x is ratio of number of 0/number of 1
            position0=random.sample(list(df_query[df_query['relevancy']==0].index),num)
            #return the sup-sample
            df_query=df_query.loc[(position1+position0)]    
        df_output=pd.concat([df_output,df_query])
        
        #df_output=df_output.loc[:,['queries','passage','relevancy']]   
    return(df_output)

df_train_data=make_data_balance(df_train_data, x=3)
df_train_data.to_csv('train_sample.tsv',sep='\t',index=False,header=True) #save
df_train_data=pd.read_csv('train_sample.tsv', sep='\t',header=[0]) #load
#%%train word2vect rather than use the pretrained one(embedding procedure.1)
def train_w2v(df):

    queries, passages = list(df['queries']), list(df['passage'])
    texts = []
    texts.extend(queries)
    texts.extend(passages)
    import nltk
    from tqdm import tqdm
    tokens = []
    for text in tqdm(texts):
        value = nltk.sent_tokenize(text)
        for i in value:
            words = nltk.word_tokenize(text=i)
            tokens.append(words)

    from gensim.models import Word2Vec
    model = Word2Vec(tokens, vector_size=50, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    print("Train word2vec Finished!")

train_w2v(pd.concat([df_validation_data]))
from gensim.models import Word2Vec
model_word2vec=Word2Vec.load("word2vec.model")
model_word2vec=model_word2vec.wv

#%%use the pretrained word2vect(embedding procedure.2)

# word2vect from google https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
from gensim.models import KeyedVectors
model_word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

#%%text processing and turn word to vector

#revise the abbreviation to the whole word
dict_word_processing=defaultdict(list) 
dict_word_processing["'ve"]="have"
dict_word_processing["'ll"]="will"
dict_word_processing["'d"]="had"
dict_word_processing["'re"]="are"


#define two function to process the text to vector based on word2vec or other models

#process the text without much procedure
def text_proc(text,model,dict_word_processing,stopword):

    output_vec=[]
    for word in word_tokenize(text):
        if word in dict_word_processing:
            word=dict_word_processing[word]        

        if word in model:                          
            word_vec=model[word]
            output_vec.append(word_vec) 
                        
    if output_vec==[]:
        output_vec=np.zeros([model.vector_size],dtype=np.float32)
    else:
        output_vec=np.mean(output_vec,axis=0)
         
    return (output_vec) 



#process the text with stemming, deleting the stopwords and deleting wierd symbols
def text_proc(text,model,dict_word_processing,stopword):

    output_vec=[]
    for word in word_tokenize(text):
        if word in dict_word_processing:
            word=dict_word_processing[word]
        
        if word not in stopword:
            if  bool(re.search(r'[a-z]',word)):
                word=PorterStemmer().stem(word)
                if word in model:                          
                    word_vec=model[word]
                    output_vec.append(word_vec) 
                        
    if output_vec==[]:
        output_vec=np.zeros([model.vector_size],dtype=np.float32)
    else:
        output_vec=np.mean(output_vec,axis=0)
         
    return (output_vec) 

#%%store the vector for text into the dictionary
#summary the unqiue passage with its pid like df_train_test
df_train_doc=df_train_data.loc[:,['pid','passage']] 
df_train_doc.drop_duplicates(subset=None, keep='first', inplace=True)
#summary the unqiue passage with its id 
df_valid_doc=df_validation_data.loc[:,['pid','passage']] 
df_valid_doc.drop_duplicates(subset=None, keep='first', inplace=True)


#form the vector and each query/passage and store it in form of dictionary
def vect_proc(df,id_name,term_name,model,stopword):
    
    output_dict=defaultdict(list)
    
    id_index=df.columns.get_loc(id_name)
    term_index=df.columns.get_loc(term_name)
    for i,id in enumerate(df[id_name]):
        if i%1000 ==0:
            print("Processing progress is %s '%%'" %(100*i/len(df.index))) 
        output_dict[df.iloc[i,id_index]]=text_proc(df.iloc[i,term_index],model,dict_word_processing,stopword)
    
    return output_dict
#%% constrcut the query/doc vector for train dataset
dict_word2vect_query_train=vect_proc(df_train_test,'qid','queries',model_word2vec,stopword)
dict_word2vect_doc_train=vect_proc(df_train_doc,'pid','passage',model_word2vec,stopword)
#%% constrcut the query/doc vector for validation dataset
dict_word2vect_query_valid=vect_proc(df_validation_test,'qid','queries',model_word2vec,stopword)
dict_word2vect_doc_valid=vect_proc(df_valid_doc,'pid','passage',model_word2vec,stopword)   
#%%save the dictionary
np.save('dict_word2vect_query_train.npy',dict_word2vect_query_train)
np.save('dict_word2vect_doc_train.npy',dict_word2vect_doc_train)
np.save('dict_word2vect_query_valid.npy',dict_word2vect_query_valid)
np.save('dict_word2vect_doc_valid.npy',dict_word2vect_doc_valid)
#%%load the dictionary
dict_word2vect_query_train=np.load('dict_word2vect_query_train.npy',allow_pickle=True).item()
dict_word2vect_doc_train=np.load('dict_word2vect_doc_train.npy',allow_pickle=True).item()
dict_word2vect_doc_valid=np.load('dict_word2vect_doc_valid.npy',allow_pickle=True).item()
dict_word2vect_query_valid=np.load('dict_word2vect_query_valid.npy',allow_pickle=True).item()
#%%def the function to construct the tensor for trainning
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable

def const_tensor(df,dict_doc,dict_query,ncol,BM25):

    qid_loc=df.columns.get_loc('qid')
    pid_loc=df.columns.get_loc('pid')
    BM25_loc=df.columns.get_loc('BM25_sc')
    
    if BM25==True:
    
        output_tensor=torch.empty([len(df),(ncol+1)])
        for i in range(len(df)):
            if i %10000==0:
                print("iteration = %s"% i)
            vect1=dict_query[df.iloc[i,qid_loc]]      
            vect2=dict_doc[df.iloc[i,pid_loc]]      
            BM25=df.iloc[i,BM25_loc]
            #concatenate the vectors
            combined_vect=np.concatenate((vect1,vect2,BM25),axis=None)
            output_tensor[i]=torch.from_numpy(combined_vect)
      
    else:    
    
        output_tensor=torch.empty([len(df),ncol])
        for i in range(len(df)):
            if i %10000==0:
                print("iteration = %s"% i)            
            vect1=dict_query[df.iloc[i,qid_loc]]      
            vect2=dict_doc[df.iloc[i,pid_loc]]      
            #concatenate the vectors
            combined_vect=np.concatenate((vect1,vect2),axis=None)
            output_tensor[i]=torch.from_numpy(combined_vect)
    return(output_tensor)
#%%construct the feature variable tensor
valid_tensor=const_tensor(df_validation_data,dict_word2vect_doc_valid,dict_word2vect_query_valid,600,BM25=True)    
train_tensor=const_tensor(df_train_data,dict_word2vect_doc_train,dict_word2vect_query_train,600,BM25=True)  
#%%save the tensor if needed
torch.save(train_tensor, "train_tensor.pt")
torch.save(valid_tensor, "valid_tensor.pt")
#%%construct the label tensor
valid_tensor_label=torch.from_numpy(np.array(list(df_validation_data['relevancy']), dtype=np.float32)).unsqueeze(1)
train_tensor_label=torch.from_numpy(np.array(list(df_train_data['relevancy']), dtype=np.float32)).unsqueeze(1)
#%%save the tensor if needed
torch.save(train_tensor_label, "train_tensor_label.pt")
torch.save(valid_tensor_label, "valid_tensor_label.pt")
#%%load the tensor if needed
valid_tensor_label=torch.load("valid_tensor_label.pt")
train_tensor_label=torch.load("train_tensor_label.pt")
valid_tensor=torch.load("valid_tensor.pt")
train_tensor=torch.load("train_tensor.pt")

#%% construct the logistic regression model 
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable

#need to decide which x_train to use
x_train=train_tensor #feature variable
x_train=train_tensor[:,0:-1] #feature variable without BM25
y_train=train_tensor_label #label
#%%
#consider to use cos-similarity as input
x1=x_train[:,0:300]
x2=x_train[:,300:]
x_train=torch.cosine_similarity(x1, x2, dim=1)
x_train=x_train.unsqueeze(1)
#re-add the BM25 if needed
x_train=torch.cat((x_train,train_tensor[:,-1].unsqueeze(1)), 1)
#%%only for part 3 (distance term)
x_train_dist=torch.nn.PairwiseDistance(2,keepdim=True).forward(x1,x2)
x_train=torch.cat((x_train,x_train_dist),1)
#%%
#add the bias term
x_train=torch.cat((x_train,torch.ones(x_train.size()[0],1)), 1)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(x_train.size()[1], 1)
        self.sm = nn.Sigmoid()
    def forward(self, x):
        out = self.lr(x)
        #print(x)
        out = self.sm(out)
        return out


#%%use gpu to train
logistic_model = LogisticRegression().cuda()
x_train = Variable(x_train).cuda()
y_train = Variable(y_train).cuda()
#%%use cpu to train
logistic_model = LogisticRegression()
x_train = Variable(x_train)
y_train = Variable(y_train)
# def loss function and optimizer
criterion = nn.BCELoss()      # def loss function
optimizer = optim.SGD(logistic_model.parameters(), lr=0.1, momentum=0.9)

loss_vec=[]
for epoch in range(2000):

    # forward procedure
    out = logistic_model(x_train)
    loss = criterion(out, y_train)

    # backward procedure
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_vec+=[loss.cpu().detach().numpy().tolist()]


#%% save the model
model_LR=logistic_model
#save the model if needed
torch.save(model_LR, "model_LR.pt")
#%%plot the loss
#by setting different learning rate, loss vector is saved
loss_lr_1=loss_vec #store loss vector when setting lr to be 1
loss_lr_0_1=loss_vec #store loss vector when setting lr to be 0.1
loss_lr_0_01=loss_vec #store loss vector when setting lr to be 0.01
loss_lr_0_001=loss_vec #store loss vector when setting lr to be 0.001

plt.title('Plot of loss')
plt.plot(loss_lr_1, color='red', label='learning rate=1')
plt.plot(loss_lr_0_1, color='blue', label='learning rate=0.1')
plt.plot(loss_lr_0_01, color='green', label='learning rate=0.01')
plt.plot(loss_lr_0_001, color='yellow', label='learning rate=0.001')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
#%%predict the result
#note that the input here for prediction must be corresponding to the input for training above

#need to decide which x_test should be used
x_test=valid_tensor #feature variable
x_test=valid_tensor[:,0:-1] #feature variable without BM25

#%%
#consider to use cos-similarity as input
x_test1=x_test[:,0:300]
x_test2=x_test[:,300:]
x_test=torch.cosine_similarity(x_test1, x_test2, dim=1)
x_test=x_test.unsqueeze(1)
#re-add BM25 if needed
x_test=torch.cat((x_test,valid_tensor[:,-1].unsqueeze(1)), 1)
#%%
#only for part 3 (distance term)
x_test_dist=torch.nn.PairwiseDistance(2,keepdim=True).forward(x_test1,x_test2)
x_test=torch.cat((x_test,x_test_dist),1)
#%%
#add the bias term
x_test=torch.cat((x_test,torch.ones(x_test.size()[0],1)), 1)
#predict the result
LR=np.concatenate(model_LR.forward(Variable(x_test)).data.numpy(),axis=None).tolist()
#return the LR score value to the data set
df_validation_data['LR']=LR
#%% calcualte the ap/ndcg under different queries

df_validation_test['LR_ndcg']='NA'
df_validation_test['LR_av_precision']='NA'
for i,qid in enumerate(df_validation_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_validation_test.index)))   
    df_query=df_validation_data[df_validation_data['qid']==qid]
    df_query=df_query.sort_values(by=['LR'], ascending=False)
    
    #calculate average precision/ndcg for each query
    df_validation_test.loc[df_validation_test['qid']==qid,'LR_ndcg']=ndcg(df_query['relevancy'],100)
    df_validation_test.loc[df_validation_test['qid']==qid,'LR_av_precision']=av_precision(df_query['relevancy'],100)

plot_acc(df_validation_test,['LR_av_precision','Random_av_precision','BM25_av_precision'],'av_precision','Plot of accuracy (trained by validation)')
#%% save the final result

def save(df,score_name,algo_name,save_name):   
    df_save=pd.DataFrame()
    qid_list=list(set(df['qid']))   #unique qid list
    
    for i,qid in enumerate(qid_list):
        df_query_score=df[df['qid']==qid]
        #rank the dataframe by the score
        df_query_score=df_query_score.sort_values(by=[score_name], ascending=False)        
        #output the top100 passages
        if len(df_query_score.index)>=100:
            df_query_score=df_query_score.iloc[0:100,:] 
        #add the rank column
        df_query_score['rank']=list(range(1,(len(df_query_score.index)+1)))  
        #add algoname2
        df_query_score['algoname2']=algo_name
        #decide the final output
        df_query_score=df_query_score.loc[:,['qid','assignment','pid','rank',score_name,'algoname2']]
        #add the temporary df to the final one
        df_save = pd.concat([df_save,df_query_score])    
    df_save.to_csv(save_name,sep='\t',index=False,header=False)

save(df_validation_data,score_name='LR',algo_name='LR',save_name='LR_1.txt')

#%%cos-similarity score is added to data as it may be useful for part3

df_train_data['cos_sim']=x_train[:,0].data.numpy().tolist()
df_validation_data['cos_sim']=x_test[:,0].data.numpy().tolist()

df_train_data.to_csv('train_sample.tsv',sep='\t',index=False,header=True) #save
#%%


PART3



#%%
#define a function calculate the number of terms in each block(prepare for Dmatrix)
def seperate(df):
    list_query=list(set(df['qid']))
    list_query.sort(key=list(df['qid']).index)
    seperation=[]
    for qid in list_query:
        df_query=df[df['qid']==qid]
        seperation+=[len(df_query)]
        
    return(np.array(seperation))

sep_train=seperate(df=df_train_data)
sep_valid=seperate(df=df_validation_data)
#%% def a function to calculate the number of words of text
def num_word(text_list):
    number_list=[]
    for text in text_list:
        number_list+=[len(word_tokenize(text))]
    return(number_list)

#calculate nwords for query/doc in train/test
train_query_nword=np.expand_dims(num_word(list(df_train_data['queries'])), axis=1)
train_doc_nword=np.expand_dims(num_word(list(df_train_data['passage'])), axis=1)
test_query_nword=np.expand_dims(num_word(list(df_validation_data['queries'])), axis=1)
test_doc_nword=np.expand_dims(num_word(list(df_validation_data['passage'])), axis=1)
#%% decide for input
import pandas as pd
import numpy as np
from xgboost import DMatrix,train
#input for train
dtrain=x_train.data.numpy() #contain BM25, cos-sim, dist
dtrain=np.concatenate((dtrain,train_query_nword),axis=1) #add number of words in query/doc
dtrain=np.concatenate((dtrain,train_doc_nword),axis=1)

dtarget=y_train.data.numpy().flatten()
xgbTrain = DMatrix(dtrain, label = dtarget)
xgbTrain.set_group(sep_train)

#input for validation(test)
dtest=x_test.data.numpy()  #contain BM25, cos-sim, dist
dtest=np.concatenate((dtest,test_query_nword),axis=1)
dtest=np.concatenate((dtest,test_doc_nword),axis=1)

xgbTest = DMatrix(dtest)
xgbTest.set_group(sep_valid)

#input for evaluation
sep_eval=sep_valid[0:150]
deval=dtest[0:sum(sep_eval),:]
xgbEval = DMatrix(deval, label = valid_tensor_label.data.numpy().flatten().tolist()[0:sum(sep_eval)])
xgbEval.set_group(sep_eval)
eval_input = [(xgbEval, 'eval')] 
#%%default model

xgb_param_plain ={    
    'booster' : 'gbtree',
    'objective' : 'rank:pairwise',
    'eval_metric' : 'ndcg@100',
}
LM_model_plain = train(xgb_param_plain,xgbTrain,num_boost_round=100,early_stopping_rounds=15,evals=eval_input,
                  evals_result=eval_store)

LM_plain=LM_model_plain.predict(xgbTest).tolist() #predict the final result
df_validation_data['LM_plain']=LM_plain

#%%parameter tuning
accuracy_table=np.zeros([3,4,4,3]) grid search
eta_list=[0.1,0.3,0.6]
gamma_list=[0,5,10,20]
min_child_weight_list=[0,0.1,1,5]
max_depth_list=[4,6,8]

for i,eta_value in enumerate(eta_list): 
    for j,gamma_value in enumerate(gamma_list): 
        for k,min_child_weight_value in enumerate(min_child_weight_list): 
            for l,max_depth_value in enumerate(max_depth_list): 
                
                eval_store={}#store the evaluation
                xgb_param ={    
                    'booster' : 'gbtree',
                    'eta': eta_value,
                    'gamma' : gamma_value ,
                    'min_child_weight' : min_child_weight_value,
                    'objective' : 'rank:pairwise',
                    'eval_metric' : 'ndcg@100',
                    'max_depth' : max_depth_value,
                }
                LM_model = train(xgb_param,xgbTrain,num_boost_round=100,early_stopping_rounds=15,evals=eval_input,
                                  evals_result=eval_store)
                accuracy_table[i,j,k,l]=eval_store['eval']['ndcg@100'][-1] #take the last(optimal) value of evaluation vector

#%%see and pick the optimal parameter combination
index1,index2,index3,index4=np.where(accuracy_table == np.max(accuracy_table))
eta_final=eta_list[index1[0]]
gamma_final=gamma_list[index2[0]]
min_child_weight_final=min_child_weight_list[index3[0]]
max_depth_final=max_depth_list[index4[0]]

#take the optimal model
xgb_param_final ={    
    'booster' : 'gbtree',
    'eta': eta_final,
    'gamma' : gamma_final ,
    'min_child_weight' : min_child_weight_final,
    'objective' : 'rank:pairwise',
    'eval_metric' : 'ndcg@100',
    'max_depth' : max_depth_final,
}
LM_model_final = train(xgb_param_final,xgbTrain,num_boost_round=100,early_stopping_rounds=15,evals=eval_input,
                  evals_result=eval_store)

LM=LM_model_final.predict(xgbTest).tolist() #predict the final result
df_validation_data['LM']=LM
#%%
#check the importance
print(LM_model_final.get_score(importance_type='gain'))
importance_value=LM_model_final.get_score(importance_type='gain')
importance_value=list(importance.values())/np.sum(list(importance.values()))*100
#replace the name 
importance_name=LM_model_final.get_score(importance_type='gain')
importance_name['f0']='Cos-similarity'
importance_name['f1']='BM25'
importance_name['f2']='Euclidean distance'
importance_name['f3']='Number of words in query'
importance_name['f4']='Number of words in document'

#plot the importance
import matplotlib.pyplot as plt
 
plt.barh(range(len(importance_value)), importance_value,tick_label = list(importance_name.values()))
plt.xlabel('Percentage of importance')
plt.title('Importance plot')
plt.show()

#%%

#plot the model with default parameter
df_validation_test['LM_plain_ndcg']='NA'
df_validation_test['LM_plain_av_precision']='NA'
for i,qid in enumerate(df_validation_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_validation_test.index)))   
    df_query=df_validation_data[df_validation_data['qid']==qid]
    df_query=df_query.sort_values(by=['LM_plain'], ascending=False)
    
    #calculate average precision/ndcg for each query
    df_validation_test.loc[df_validation_test['qid']==qid,'LM_plain_ndcg']=ndcg(df_query['relevancy'],100)
    df_validation_test.loc[df_validation_test['qid']==qid,'LM_plain_av_precision']=av_precision(df_query['relevancy'],100)

#plot the model with optimal parameter
df_validation_test['LM_ndcg']='NA'
df_validation_test['LM_av_precision']='NA'
for i,qid in enumerate(df_validation_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_validation_test.index)))   
    df_query=df_validation_data[df_validation_data['qid']==qid]
    df_query=df_query.sort_values(by=['LM'], ascending=False)
    
    #calculate average precision/ndcg for each query
    df_validation_test.loc[df_validation_test['qid']==qid,'LM_ndcg']=ndcg(df_query['relevancy'],100)
    df_validation_test.loc[df_validation_test['qid']==qid,'LM_av_precision']=av_precision(df_query['relevancy'],100)

plot_acc(df_validation_test.iloc[150:],['LM_av_precision','LM_plain_av_precision','Random_av_precision','BM25_av_precision'],'av_precision','Plot of accuracy (trained by balanced data)')
#%% save the final output
save(df_validation_data,score_name='LM',algo_name='LM',save_name='LM.txt')

#%%


PART4



#%% Double DNN +cos-similarity
#load the tensor constructed in part2
y_tensor_test=torch.load("valid_tensor_label.pt")
y_tensor_train=torch.load("train_tensor_label.pt")
x_tensor_test=torch.load("valid_tensor.pt")[:,0:-1]
x_tensor_train=torch.load("train_tensor.pt")[:,0:-1]

x1_train=x_tensor_train[:,0:300]
x2_train=x_tensor_train[:,300:]

x1_test=x_tensor_test[:,0:300]
x2_test=x_tensor_test[:,300:]

#%%
# Neural Network parameters


l_r = 0.01
dropout = torch.nn.Dropout(p=0.1)

Layer1_size=256
Layer2_size=int(Layer1_size/4)
Layer3_size=int(Layer2_size/8)
Layer4_size=int(Layer3_size/16)


#Neural Network layers
linear_layer1=torch.nn.Linear(300, Layer1_size, bias=True) 
linear_layer2=torch.nn.Linear(Layer1_size, Layer2_size)
linear_layer3=torch.nn.Linear(Layer2_size, Layer3_size)
linear_layer4=torch.nn.Linear(Layer3_size, Layer4_size)

#active function
sigmoid = torch.nn.Sigmoid()
tanh=torch.nn.Tanh()
relu=torch.nn.LeakyReLU()
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#%%

#Neural network architecture for query
network1 = torch.nn.Sequential(linear_layer1,nn.BatchNorm1d(Layer1_size),relu,
                          linear_layer2,dropout,relu,
                          linear_layer3,dropout,relu,
                          #linear_layer4,dropout,relu,
                          )
#Neural network architecture for document
network2 = torch.nn.Sequential(linear_layer1,nn.BatchNorm1d(Layer1_size),relu,
                          linear_layer2,dropout,relu,
                          linear_layer3,dropout,relu,
                          #linear_layer4,dropout,relu,
                          )



#%%
optimizer1.zero_grad() #clean the gradient
optimizer2.zero_grad()
optimizer1 = torch.optim.Adam(network1.parameters(), lr=l_r,weight_decay=1e-2)
optimizer2 = torch.optim.Adam(network2.parameters(), lr=l_r,weight_decay=1e-2)
loss_function=torch.nn.BCELoss()
epochs = 200
loss_vector= []

#Training in batches
for epoch in range(epochs):    
    output=torch.abs(cos(network1(x1_train),network2(x2_train)).unsqueeze(dim=1))
    loss = loss_function(output, y_tensor_train) 
    loss.backward()         # backpropagation
    optimizer1.step()        # apply gradients 
    optimizer2.step()
    
    if epoch % 10 == 0:        
        loss = loss.data
        loss_vector.append(loss)
        print(epoch, loss.data.cpu().numpy())

#%% calculate the predictiion score

NN=(torch.abs(cos(network1(x1_test),network2(x2_test)).unsqueeze(dim=1)).data) #predict the final result
NN=np.concatenate(NN.data.numpy(),axis=None).tolist()
df_validation_data['NN']=NN
#%%

df_validation_test['NN_ndcg']='NA'
df_validation_test['NN_av_precision']='NA'
for i,qid in enumerate(df_validation_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_validation_test.index)))   
    df_query=df_validation_data[df_validation_data['qid']==qid]
    df_query=df_query.sort_values(by=['NN'], ascending=False)
    
    #calculate average precision/ndcg for each query
    df_validation_test.loc[df_validation_test['qid']==qid,'NN_ndcg']=ndcg(df_query['relevancy'],100)
    df_validation_test.loc[df_validation_test['qid']==qid,'NN_av_precision']=av_precision(df_query['relevancy'],100)


sum(df_validation_test['NN_av_precision']==0)
plot_acc(df_validation_test,['NN_av_precision','BM25_av_precision','Random_av_precision'],'av_precision','Plot of accuracy (1 layers)')

#%%

#For DSSM (run in the project)


#%%
#The result from real DSSM

df_validation_data=pd.read_csv('result.csv', sep=',',header=[0]) 

df_validation_test['NN_ndcg']='NA'
df_validation_test['NN_av_precision']='NA'
for i,qid in enumerate(df_validation_test['qid']):
    
    print("Processing progress is %s '%%'" %(100*i/len(df_validation_test.index)))   
    df_query=df_validation_data[df_validation_data['qid']==qid]
    df_query=df_query.sort_values(by=['cos_score'], ascending=False)
    
    #calculate average precision/ndcg for each query
    df_validation_test.loc[df_validation_test['qid']==qid,'NN_ndcg']=ndcg(df_query['relevancy'],100)
    df_validation_test.loc[df_validation_test['qid']==qid,'NN_av_precision']=av_precision(df_query['relevancy'],100)


sum(df_validation_test['NN_av_precision']==0)
plot_acc(df_validation_test,['NN_av_precision','BM25_av_precision','Random_av_precision'],'av_precision','Plot of accuracy (By dssm)')

save(df_validation_data,score_name='cos_score',algo_name='NN',save_name='NN.txt')

