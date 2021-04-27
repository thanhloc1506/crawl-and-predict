
from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
from numpy import genfromtxt
import numpy as np
import re
import tkinter as tk
import mysql.connector
# train data
def getIdx(key, arr):
    for i in range(len(arr)):
        if key == arr[i]:
            return i
def makeTrainData(_input, dic):
    output = np.zeros((1,180), dtype = int)
    count = 0
    for row in output:
        text = _input[count]
        count += 1
        _text = text.split(' ')
        for i in _text:
            if str(i) in dic:
                idx = getIdx(i, dic)
                row[idx] += 1
    return output


# creating main tkinter window/toplevel 
master = tk.Tk()
master.title("Crawl Data Vjpr0")
master.config(height=5000, width=5000)
W="e"
# this will create a label widget 
l1 = tk.Label(master, text = "Content",width=10) 
l2 = tk.Label(master, text = "Kind",width=10) 
  
# grid method to arrange labels in respective 
# rows and columns as specified 
l1.grid(row = 0, column = 0, sticky = W, pady = 3) 
l2.grid(row = 1, column = 0, sticky = W, pady = 3) 
  
# Input 
i1 = tk.Text(master,height=3,width=30) 
i2 =tk.Text(master,height=3,width=30) 
  
i1.grid(row = 0, column = 1, pady =3) 
i2.grid(row = 1, column = 1, pady = 3) 
  


#Out put


o1=tk.Text(master,height=5,width=65,bg='light cyan')
o1.grid(row = 0, column = 4, padx = 30,pady=30) 

o2=tk.Text(master,height=15,width=65,bg='light cyan')
o2.grid(row = 1, column = 4, padx = 30,pady=30) 


  
# button widget 
b1 = tk.Button(master, text = "Predict",height=2,width=20) 
b2 = tk.Button(master, text = "Find",height=2,width=20) 
  
# arranging button widgets 
b1.grid(row = 0, column = 2, sticky = W, padx = 30,pady=30) 
b2.grid(row = 1, column = 2, sticky = W, padx = 30,pady=30) 
b1.config(anchor='center')



def print_text():
    text=i1.get("1.0",tk.END)
    o1.insert(tk.INSERT,text)


class number():
    def __init__(self):
        self.num=0


s=number()



def getData(theloai):
    mydb = mysql.connector.connect(
        host="localhost",
        user="user",
        password="thanhloc0160",
        database="baotuoitre"
    )
    mycursor = mydb.cursor()
    theloai=theloai.split('\n')[0]
    query=f"SELECT * FROM tuoitre where kind = '{theloai}'"
    mycursor.execute(query)

    myresult = mycursor.fetchall()
    data = []
    for i in myresult:
        data.append({
            'title': i[0],
            'abstract': i[1],
            'content': i[2],
            'theloai': i[3],
            'image': i[4]
        })
    
    return data


def loop(context,kind):
    text=''
    data = getData(kind)
    for i in range(s.num,s.num+1):
        if (i>=len(data)): break
        text+='Title: ' + str(data[i]['title']) + '\n' \
            + '-----------------------------------------------------------------' \
            + 'Abstract: ' + str(data[i]['abstract']) + '\n' \
            + '-----------------------------------------------------------------' \
            + 'Content: ' + str(data[i]['content']) + '\n' \
            + '-----------------------------------------------------------------' \
            + 'Image_url: ' + str(data[i]['image']) + '\n' 
    context.delete("1.0",tk.END)
    context.insert(tk.INSERT, text)
    s.num+=1
    
def getDic():
    dic = ['12', 'an', 'báo', 'chủ', 'công', 'dân', 'giao', 'hcm', 'hội', 'nam', 'người', 'nhân', 'phòng', 'quốc', 'thông',
    'tp', 'tra', 'trung', 'tỉnh', 'văn', 'vụ', 'xe', 'xã', 'đường'] + \
    ['công', 'doanh', 'dịch', 'gia', 'giá', 'hàng', 'hội', 'kinh', 'nam', 'nghiệp', 'phát', 'quốc', 'thị', 
    'tp', 'trường', 'ty', 'tư', 'tỉ', 'việt', 'đầu', 'đồng', 'động'] + \
    ['chủ', 'chức', 'công', 'gia', 'hà', 'học', 'hội', 'một', 'nam', 'nghệ', 'nguyễn', 'những', 'nội', 
    'sách', 'sĩ', 'thành', 'tp', 'tác', 'tổ', 'việt', 'văn', 'ảnh'] + \
    ['ca', 'công', 'diễn', 'giả', 'giải', 'hình', 'một', 'nam', 'ng', 'nghệ', 'nhân', 'nhạc', 'phim', 'phát', 
    'quốc', 'sĩ', 'thành', 'thông', 'trong', 'viên', 'việt', 'ảnh'] + \
    ['công', 'dục', 'gd', 'giáo', 'hcm', 'học', 'nghiệp', 'quốc', 'sinh', 'thi', 'thí', 'tp', 'trường', 
    'tuyển', 'tế', 'tổ', 'viên', 'văn', 'đh', 'điểm', 'đt', 'đại'] + \
    ['11', '12', '19', 'bang', 'biden', 'báo', 'bầu', 'các', 'công', 'cử', 'dịch', 'hãng', 'mỹ', 'phiếu', 'phát',
    'quốc', 'thông', 'thống', 'trong', 'trump', 'trung', 'tổng', 'ông'] + \
    ['báo', 'bão', 'bắc', 'bộ', 'các', 'có', 'công', 'cứu', 'dự', 'gia', 'học', 'khoa', 'khu', 'khí', 'mưa', 'nam', 
    'phát', 'quốc', 'trong', 'trung', 'tâm', 'đông', 'độ'] + \
    ['chiếc', 'công', 'giao', 'giá', 'hàng', 'hành', 'lái', 'mẫu', 'nam', 'quốc', 'sản', 'thông', 'thị', 
    'triệu', 'trường', 'việt', 'xe', 'xuất', 'ôtô', 'đường', 'đồng', 'động']
    return dic


    

def predict_label(input_text=""):
    input_text=i1.get("1.0",tk.END)
    input_text = input_text.lower()
    # print(input_text)
    train_data = genfromtxt('./csv/matrix_train.csv', delimiter=',', dtype= int)
    label = genfromtxt('./csv/label.csv', delimiter=',', dtype= str)
    test_data = genfromtxt('./csv/matrix_test.csv', delimiter=',', dtype= int)
    _dic = getDic()
    
    newstr = input_text.replace("\"", "")
    newstr = newstr.replace(',', '')
    newstr = newstr.replace('.', '')
    newstr = newstr.replace('!', '')
    newstr = newstr.replace('\n', '')
    clean_input = [newstr.replace("\'", "")]
    print(clean_input)
    input_data = makeTrainData(clean_input, _dic)
    print(input_data)

    # test data
    d5 = [input_data[0]]
    # d6 = [input_data[1]]
    ## call MultinomialNB
    model = MultinomialNB()
    # training
    model.fit(train_data, label)
    # test
    text1=f'Predicting class of this text: {str(model.predict(d5)[0])} \n\
Probability of this text in each class: \n\
    giaitri: {round(model.predict_proba(d5)[0][0], 3)*100}%\n\
    giaoduc: {round(model.predict_proba(d5)[0][1], 3)*100}%\n\
    khoahoc: {round(model.predict_proba(d5)[0][2], 3)*100}%\n\
    kinhdoanh: {round(model.predict_proba(d5)[0][3], 3)*100}%\n\
    thegioi: {round(model.predict_proba(d5)[0][4], 3)*100}%\n\
    thoisu: {round(model.predict_proba(d5)[0][5], 3)*100}%\n\
    vanhoa: {round(model.predict_proba(d5)[0][6], 3)*100}%\n\
    xe: {round(model.predict_proba(d5)[0][7], 3)*100}%'



    o1.delete("1.0",tk.END)
    o1.insert(tk.INSERT, text1)

b1['command']=lambda: predict_label()

b2['command']=lambda: loop(o2,i2.get("1.0",tk.END))

# infinite loop which can be terminated  
# by keyboard or mouse interrupt 
# print_text()
tk.mainloop() 



