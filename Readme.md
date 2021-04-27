# Installation

## Package required

```bash
pip install requests
```

```bash
pip install -–upgrade pip
```

```bash
pip install Pillow
```

```bash
pip install beautifulsoup4
```

# Run

```bash
Python craw.py
```

# Database
```bash
Đã thêm dữ liệu crawl được vô database bằng hàm storeDB
Nhưng phải connect với mysql theo format
mydb = mysql.connector.connect(
     host="localhost",
     user="myuser",
     password="mypassword",
     database="mydatabase"
)
```
