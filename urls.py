class urls:
    def __init__(self):
        self.lists = []

    def store_url(self):
        self.lists = [
                    #   "https://tuoitre.vn/", 
                      "https://tuoitre.vn/thoi-su/trang",
                      "https://tuoitre.vn/kinh-doanh/trang", 
                      "https://tuoitre.vn/van-hoa/trang", 
                      "https://tuoitre.vn/giai-tri/trang", 
                      "https://tuoitre.vn/giao-duc/trang", 
                      "https://tuoitre.vn/the-gioi/trang",
                      "https://tuoitre.vn/khoa-hoc/trang", 
                      "https://tuoitre.vn/xe/trang", 



                    #   "https://tuoitre.vn/suc-khoe/trang", 
                    #   "https://tuoitre.vn/gia-that.htm",
                    #   "https://tuoitre.vn/ban-doc-lam-bao.htm", 
                    #   "https://tuoitre.vn/can-biet.htm",
                      ]
        return self.lists
    def getString(self, s):
        link1 = 'https://tuoitre.'
        if (s == 'https://tuoitre.vn/'):
            result = 'tuoitre'
            return result
        elif (s == 'https://tuoitre.vn/tien-toi-dai-hoi-dang-toan-quoc-e619'):
            result = 'daihoidang'
            return result
        elif (s[0:16:1] == link1):
            result = s[19:len(s) - 4:1]
            result = result.replace('-', '')
            return result
        else:
            result = s[8:len(s) - 12:1]
            return result