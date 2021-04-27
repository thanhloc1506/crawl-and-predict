import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="LocT@2031",
    database="baotuoitre"
)

mycursor = mydb.cursor()


def createTable():
    mycursor.execute(
        "CREATE TABLE TuoiTre (title varchar(255) PRIMARY KEY, abstract LONGTEXT, content LONGTEXT, \
        theloai LONGTEXT, image LONGTEXT)")
    mycursor.execute("SHOW TABLES")


def storedDB(title, abstract, content, theloai, image, TABLE):

    mydb = mysql.connector.connect(
        host="localhost",
        user="user",
        password="LocT@2031",
        database="baotuoitre"
    )
    # print(mydb)
    mycursor = mydb.cursor()
    sql = "INSERT INTO tuoitre (title, abstract, content, theloai,image) VALUES (%s, %s, %s, %s, %s)"
    val = (title, abstract, content, theloai, image)
    mycursor.execute(sql, val)
    mydb.commit()


def query(theloai):
    mydb = mysql.connector.connect(
        host="localhost",
        user="user",
        password="LocT@2031",
        database="baotuoitre"
    )
    mycursor = mydb.cursor()
    mycursor.execute(f"SELECT * FROM tuoitre where theloai = '{theloai}'")
    myresult = mycursor.fetchall()
    for x in myresult:
        print(x, "\n")


if __name__ == "__main__":
    createTable()
