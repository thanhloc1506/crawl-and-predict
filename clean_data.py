'''
Classification article
Case study: Bag Of Words, TF-IDF
Model: 
    - Linear classifiers
    - Decision Tree
@author: Ngo Quoc Khanh
@team: Crewmate
@email: khanh.ngoakatekhanh@hcmut.edu.vn
@Student ID: 1812593
'''
############################################
''' 
Library required
    - nltk
    - pandas and numpy
    - sklearn
'''
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
from itertools import islice
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

file_stopword = './stop_word.txt'
file_in = './process_data/article_preprocessored.csv'
file_out = './process_data/article_processored.csv'
file_test = './process_data/test.csv'
normalize_file = './process_data/article_normalized_test.csv'
# Create data sample
def text_to_csv(file_in, file_out):
    with open(file_in, 'r', encoding='ISO-8859-1') as f1, open(file_out, 'w') as f2:
        f2.write('Id,theloai,Title,content\n')
        
# Read file and review datas
def reviews_data(in_file):
    if os.path.exists(in_file):
        print('=== Read data from', in_file, '===')   
        frame = pd.read_csv(in_file)
        print(frame.shape, len(frame))
        return frame

#frame = reviews_data(file_in)

def vietnamese_stopwords(file_name):
    with open(file_name,'r') as f1:
        lines = f1.readlines()
        stop_words_1 = []
        stop_words_2 = []
        for line in lines:
            if len(line.strip('\n').split()) == 1:
                stop_words_1.append(line.strip('\n'))
            else: stop_words_2.append(line.strip('\n'))
        
        return stop_words_1, stop_words_2
    
stop_words_1, stop_words_2 = vietnamese_stopwords(file_stopword)        
    
def clean_text(sentence):
    #1. Remove non letter
    words = re.sub('[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]',' ',sentence)
    #2. To lowercase 
    words = sentence.lower()
    #3. Join them together
    #return ' '.join(words)
    return words

def normalize(file_in):
    data_frame = reviews_data(file_in)
    data_frame = data_frame.iloc[:,1:4]
    # Vietnamese stop words
    stop_words_1, stop_words_2 = vietnamese_stopwords(file_stopword)
    #Remove no information value and clean text
    for ind in data_frame.index:
        if data_frame['title'][ind] == '' or data_frame['content'][ind] == '':
            data_frame = data_frame.drop(ind, axis=0)
        else:
            # Clean raw text
            data_frame['theloai'][ind] = clean_text(data_frame['theloai'][ind])
            data_frame['title'][ind] = clean_text(data_frame['title'][ind])
            data_frame['content'][ind] = clean_text(data_frame['content'][ind])    
            # Remove stop words "Lon hon mot am tiet"
            
            for word in stop_words_2:
                if word in data_frame['title'][ind]: 
                    data_frame['title'][ind] = data_frame['title'][ind].replace(word, '')
                if word in data_frame['content'][ind]: 
                    data_frame['content'][ind] = data_frame['content'][ind].replace(word, '')    
            
            #Remove stop words "Tu mot am tiet"
            data_frame['theloai'][ind] = data_frame['theloai'][ind].split()
            data_frame['title'][ind] = data_frame['title'][ind].split()
            data_frame['content'][ind] = data_frame['content'][ind].split()
            for word in stop_words_1:
                if word in data_frame['theloai'][ind]: 
                    data_frame['theloai'][ind].remove(word)
                if word in data_frame['title'][ind]: 
                    data_frame['title'][ind].remove(word)
                if word in data_frame['content'][ind]: 
                    data_frame['content'][ind].remove(word)  
            
                
    return data_frame

def print_words_fre(data):
    
    vocab = vectorizer.get_feature_names()
    # Sum up the counts of each vocabulary word
    dist = np.sum(data, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print ("Words frequency...")
    for tag, count in zip(vocab, dist):
        print (count, tag)
    res = {} 
    for key,value in zip(vocab, dist):
        res[key] = value
        
    return res
def train_data(x_train, y_train, x_test, y_test):
    # list of classifier:
    classifier_name = ['Decision Tree' ]
    classifier_model = [
        DecisionTreeClassifier(max_depth = 4),
        GaussianNB()]
    
    # Empty Dictionary
    result = {}
    # train data using scikit-learn lib
    for (name, model) in zip(classifier_name, classifier_model):
        score = model.fit(X_train, Y_train).score(X_test, Y_test)
        result[name] = score
    
    #Print the Results
    print("================================")
    print("===========RESULT===============")
    print("================================")
    for name in result:
        print(name + ' : accurency ' + str(round(result[name], 4)))
    print("================================")   
    

# RUN
if __name__ == '__main__':
    '''
    data_ = normalize(file_test)
    for idx in data_.index:
        data_['theloai'][idx] = ' '.join(data_['theloai'][idx])
        data_['title'][idx] = ' '.join(data_['title'][idx])
        data_['content'][idx] = ' '.join(data_['content'][idx])
    data_.to_csv('./process_data/article_normalized_test.csv')
    '''    
    
    temp = pd.read_csv(normalize_file)
    for idx in temp.index:
        if 'giaitri' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'giaitri'
        if 'giaoduc' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'giaoduc'    
        if 'khoahoc' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'khoahoc'
        if 'kinhdoand' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'kinhdoand'
        if 'thegioi' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'thegioi'    
        if 'thoisu' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'thoisu'
        if 'vanhoa' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'vanhoa'
        if 'xe' in temp['theloai'][idx]:
            temp['theloai'][idx] = 'xe'
        
        vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=25)
        
    # Split data into train and test
    train_, test_ = train_test_split(temp, test_size=0.2)
    
    train_des = train_['content'].values.astype('U')
    train_theloai = train_['theloai'].values.astype('U')
    test_des = test_['content'].values.astype('U')
    test_theloai = test_['theloai'].values.astype('U')
    
    X_train = vectorizer.fit_transform(train_des).toarray()
    Y_train = train_theloai
    
    X_test = vectorizer.fit_transform(test_des).toarray()
    Y_test = test_theloai
    # dic = print_words_fre(X_train)
    # dic = sorted(dic.items(), key=lambda x: x[1])
    # names = [x[0] for x in dic]
    # values = [x[1] for x in dic]
    # ax = plt.plot(names, values)
    # plt.show()
    
    train_data(X_train, Y_train, X_test, Y_test)