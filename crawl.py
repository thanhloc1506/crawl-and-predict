from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import requests
from urls import urls
import json
import os.path
import mysql.connector
from databases import db
import pandas as pd
import csv
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


def reviews_data(in_file):
    if os.path.exists(in_file):
        print('=== Read data from', in_file, '===')
        frame = pd.read_csv(in_file)
        print(frame.shape, len(frame))
        return frame


def vietnamese_stopwords(file_name):
    with open(file_name, 'r') as f1:
        lines = f1.readlines()
        stop_words_1 = []
        stop_words_2 = []
        for line in lines:
            if len(line.strip('\n').split()) == 1:
                stop_words_1.append(line.strip('\n'))
            else:
                stop_words_2.append(line.strip('\n'))

        return stop_words_1, stop_words_2


stop_words_1, stop_words_2 = vietnamese_stopwords('./stop_word.txt')


def clean_text(sentence):
    # 1. Remove non letter
    words = re.sub('[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]', ' ', sentence)
    # 2. To lowercase
    words = sentence.lower()
    # 3. Join them together
    # return ' '.join(words)
    return words


def normalize(file_in):
    data_frame = reviews_data(file_in)
    data_frame = data_frame.iloc[:, 0:4]
    # print(data_frame)
    # Vietnamese stop words
    stop_words_1, stop_words_2 = vietnamese_stopwords('./stop_word.txt')
    # Remove no information value and clean text
    for ind in data_frame.index:
        data_frame['content'][ind] = str(data_frame['content'][ind])
        data_frame['title'][ind] = str(data_frame['title'][ind])
        if data_frame['title'][ind] == '' or data_frame['content'][ind] == '':
            data_frame = data_frame.drop(ind, axis=0)
        else:
            # Clean raw text
            try:
                data_frame['theloai'][ind] = clean_text(
                    data_frame['theloai'][ind])
            except:
                pass
            try:
                data_frame['title'][ind] = clean_text(data_frame['title'][ind])
            except:
                pass
            try:
                data_frame['content'][ind] = clean_text(
                    data_frame['content'][ind])
            except:
                pass

            # Remove stop words "Lon hon mot am tiet"
            for word in stop_words_2:
                if word in data_frame['title'][ind]:
                    try:
                        data_frame['title'][ind] = data_frame['title'][ind].replace(
                            word, '')
                    except:
                        pass
                if word in data_frame['content'][ind]:
                    try:
                        data_frame['content'][ind] = data_frame['content'][ind].replace(
                            word, '')
                    except:
                        pass

            # Remove stop words "Tu mot am tiet"
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


def train_data(x_train, y_train, x_test, y_test):
    # list of classifier:
    classifier_name = ['Decision Tree']
    classifier_model = [
        DecisionTreeClassifier(max_depth=4),
        GaussianNB()]

    # Empty Dictionary
    result = {}
    # train data using scikit-learn lib
    for (name, model) in zip(classifier_name, classifier_model):
        score = model.fit(X_train, Y_train).score(X_test, Y_test)
        result[name] = score

    # Print the Results
    print("================================")
    print("===========RESULT===============")
    print("================================")
    for name in result:
        print(name + ' : accurency ' + str(round(result[name], 4)))
    print("================================")


def print_words_fre(data):

    vocab = vectorizer.get_feature_names()
    # Sum up the counts of each vocabulary word
    dist = np.sum(data, axis=0)
    word_dic = []
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print("Words frequency...")
    for tag, count in zip(vocab, dist):
        word_dic += [tag]
        print(count, tag)
    res = {}
    for key, value in zip(vocab, dist):
        res[key] = value

    return res, word_dic


def storedDB(title, abstract, content, theloai, image, TABLE):

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="LocT@2031",
        database="baotuoitre"
    )
    # print(mydb)
    mycursor = mydb.cursor()
    sql = "INSERT INTO tuoitre (title, abstract, content, theloai,image) VALUES (%s, %s, %s, %s, %s)"
    val = (title, abstract, content, theloai, image)
    mycursor.execute(sql, val)
    mydb.commit()


def storeCSV(data, fileName):
    with open(fileName, mode='w') as paperCrawler_file:
        employee_writer = csv.writer(
            paperCrawler_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(
            ['title', 'abstract', 'content', 'theloai', 'image'])
        for i in range(len(data)):
            title = data[i]['title']
            abstract = data[i]['abstract']
            content = data[i]['content']
            theloai = data[i]['theloai']
            image = data[i]['image']
            employee_writer.writerow(
                [f'{title}', f'{abstract}', f'{content}', f'{theloai}', f'{image}'])


def storeCSV2(data, fileName):
    with open(fileName, mode='w') as nor_file:
        employee_writer = csv.writer(
            nor_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(
            ['title', 'abstract', 'content', 'theloai', 'image'])
        for i in range(len(data)):
            title = data['title'][i]
            abstract = data['abstract'][i]
            content = data['content'][i]
            theloai = data['theloai'][i]
            # image = data['image'][i]
            employee_writer.writerow(
                [f'{title}', f'{abstract}', f'{content}', f'{theloai}'])


def crawNewsData(baseUrl, url, count, title_page):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    titles = soup.findAll('h3', class_='title-news')
    links = [link.find('a').attrs["href"] for link in titles]
    data = []
    csv_data = []
    save_path = "./News/"
    for link in links:
        news = requests.get(baseUrl + link)
        soup = BeautifulSoup(news.content, "html.parser")
        try:
            title = soup.find("h1", class_="article-title").text
        except:
            title = ""
        try:
            abstract = soup.find("h2", class_="sapo").text
        except:
            abstract = ""
        try:
            body = soup.find("div", id="main-detail-body")
        except:
            body = ""
        content = ""
        try:
            content = body.findChildren("p", recursive=False)[
                0].text + body.findChildren("p", recursive=False)[1].text
        except:
            content = ""
        try:
            image = body.find("img").attrs["src"]
        except:
            image = ""

        try:
            comment_part = soup.find("div", class_="comment_list")
            comment = comment_part.findChildren("li", recursive=False)[0].text
        except:
            try:
                comment = comment_part.find("p").text
                comment = comment.replace("\r\n            ", "")
                comment = comment.replace("\r\n        ", "")
            except:
                comment = ""

        data.append({
            "title": title,
            "abstract": abstract,
            "content": content,
            "image": image,
            "theloai": title_page,
            "comment": comment,
        })

        if data:
            print(
                "_________________________________________________________________________")
            count += 1
            print(count)
            print("Tiêu đề: " + title)
            print("Mô tả: " + abstract)
            print("Nội dung: " + content)
            print("Ảnh minh họa: " + image)
            print("Bình luận: " + comment)
        else:
            print("Can't crawl in this page")
        try:
            storedDB(title, abstract, content, title_page, image, 'TuoiTre')
            storeCSV(title, abstract, content, title_page, image)
            pass
        except:
            print("Tiếp tục Crawl")
    # with open(save_path + title_page + '.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
    return data, count


def writeToImage(image, text, position, font, color, maxLine):
    charPerLine = 650 // font.getsize('x')[0]
    pen = ImageDraw.Draw(image)
    yStart = position[1]
    xStart = position[0]
    point = 0
    prePoint = 0
    while point < len(text):
        prePoint = point
        point += charPerLine
        while point < len(text) and text[point] != " ":
            point -= 1
        pen.text((xStart, yStart), text[prePoint:point], font=font, fill=color)
        yStart += font.getsize('hg')[1]
        maxLine -= 1
        if (maxLine == 0):
            if (point < len(text)):
                pen.text((xStart, yStart), "...", font=font, fill="black")
            break


def makeFastNews(data):
    for index, item in enumerate(data):
        # print(index, item)
        # make new image and tool to draw
        image = Image.new('RGB', (650, 750), color="white")
        pen = ImageDraw.Draw(image)
        # load image from internet => resize => paste to main image
        pen.rectangle(((0, 0), (650, 300)), fill="grey")
        newsImage = Image.open(requests.get(item["image"], stream=True).raw)
        newsImage.thumbnail((650, 300), Image.ANTIALIAS)
        image.paste(newsImage, (650 // 2 - newsImage.width //
                                2, 300 // 2 - newsImage.height // 2))
        # write title
        titleFont = ImageFont.truetype("./font/arial.ttf", 22)
        writeToImage(image, item["title"], (10, 310), titleFont, "black", 3)
        abstractFont = ImageFont.truetype("./font/arial.ttf", 15)
        writeToImage(image, item["abstract"],
                     (10, 390), abstractFont, "gray", 4)
        contentFont = ImageFont.truetype("./font/arial.ttf", 20)
        writeToImage(image, item["content"],
                     (10, 460), contentFont, "black", 11)
        name = item['theloai'] + str(index) + ".png"
        image.save("./images/" + name)
        print("saved to " + "news/" + name)


def getDic():
    _dic = ['12', 'an', 'báo', 'chủ', 'công', 'dân', 'giao', 'hcm', 'hội', 'nam', 'người', 'nhân', 'phòng', 'quốc', 'thông',
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
    return _dic


def getIdx(key, arr):
    for i in range(len(arr)):
        if key == arr[i]:
            return i


def makeTrainData(_input, dic):
    output = np.zeros((8000, 180), dtype=int)
    count = 0
    label = []
    for row in output:
        text = _input[count]['content']
        label += [_input[count]['theloai']]
        count += 1
        _text = text.split(' ')
        for i in _text:
            if i in dic:
                idx = getIdx(i, dic)
                row[idx] += 1
    pd.DataFrame(output).to_csv(
        "./csv/matrix_train.csv", header=None, index=None)
    pd.DataFrame(label).to_csv("./csv/label.csv", header=None, index=None)


def makeTestData(_input, dic):
    output = np.zeros((2000, 180), dtype=int)
    count = 0
    for row in output:
        text = _input[count]['content']
        print(text)
        count += 1
        _text = text.split(' ')
        for i in _text:
            if i in dic:
                idx = getIdx(i, dic)
                row[idx] += 1
    pd.DataFrame(output).to_csv(
        "./csv/matrix_test.csv", header=None, index=None)


_dic = getDic()

if __name__ == "__main__":
    count = 0
    numPage = 0
    page = urls().store_url()
    _urls = []
    _data = []
    _word_dic = []
    for i in range(len(page)):
        for j in range(1, 84):
            _urls += [page[i] + f"-{j}.htm"]

    print("Tim kiem theo tu khoa ----------------------------------> Press 1")
    print("Tim kiem theo chu de -----------------------------------> Press 2")
    print("Lay du lieu tu database len theo the loai --------------> Press 3")
    print("- Nhap lua chon: ", end='')
    a = int(input())
    if a == 2:
        while(count < 10000):

            title_page = urls().getString(_urls[numPage+1])
            newdata, count = crawNewsData(
                "https://tuoitre.vn", _urls[numPage], count, urls().getString(_urls[numPage]).split('/')[0])
            # makeFastNews(newdata)
            _data += newdata

            if count < 10000 and numPage < len(_urls) - 2:
                # print("\nChuyen trang ---------> Press 1")
                # print("Ket thuc crawl -------> Press 2")
                # print("- Nhap lua chon: ", end='')
                # s = input()
                s = '1'
            else:
                print("Ket thuc crawl.")
                break

            if s == '1':
                if count < 10000 and numPage < len(_urls) - 2:
                    print("Chuyen trang " + title_page + "\n")
                else:
                    print("Ket thuc Crawl.")
                numPage += 1
                if(numPage == len(_urls) - 1):
                    break
            elif s == '2':
                print("Ket thuc Crawl.")
                break
            else:
                pass
        storeCSV(_data, './csv/paperCrawler_file.csv')
        data_nor = normalize('./csv/paperCrawler_file.csv')
        storeCSV2(data_nor, './csv/nor_file.csv')
        temp = pd.read_csv('./csv/nor_file.csv')

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
        # print(_data)
        # _train, _test = train_test_split(_data, test_size = 0.2)
        # makeTrainData(_train, _dic)
        # makeTestData(_test, _dic)
        train_theloai = train_['theloai'].values.astype('U')
        test_des = test_['content'].values.astype('U')
        test_theloai = test_['theloai'].values.astype('U')

        X_train = vectorizer.fit_transform(train_des).toarray()
        Y_train = train_theloai

        X_test = vectorizer.fit_transform(test_des).toarray()
        Y_test = test_theloai
        dic, tag = print_words_fre(X_train)
        print(tag)
        dic = sorted(dic.items(), key=lambda x: x[1])

        # names = [x[0] for x in dic]
        # values = [x[1] for x in dic]
        # ax = plt.plot(names, values)
        # plt.show()

        train_data(X_train, Y_train, X_test, Y_test)

    elif a == 1:
        while count < 10000:

            print("- Nhap tu khoa: ", end='')
            field = input()
            url = 'https://tuoitre.vn/tim-kiem.htm?keywords=' + field
            title_page = urls().getString(_urls[numPage+1])
            newdata, count = crawNewsData(
                "https://tuoitre.vn", url, count, 'Ket qua cho ' + field)
            # makeFastNews(newdata)

            if count < 10000:
                print("\nTiep tuc tim theo tu khoa --------------> press 1")
                print("Ket thuc crawl -------------------------> press 2")
                print("- Nhap lua chon: ", end='')
                i = input()
                if i == '2':
                    print('Ket thuc crawl')
                    break
    elif a == 3:
        print("Nhap the loai:")
        s = input()
        db.query(s)
