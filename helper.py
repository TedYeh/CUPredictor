from bs4 import BeautifulSoup
import requests
import os

def get_image():
    for page in range(1, 213):
        response = requests.get(f"https://xslist.org/tw/listby/new/page/{page}")
        soup = BeautifulSoup(response.text, "lxml")
        
        results = soup.find_all("img")[1:]
        
        contents = soup.select('li > p')            
        
        image_links = [result.get("src") for result in results]  # 取得圖片來源連結
        
        for index, (link, content) in enumerate(zip(image_links, contents)):
            bust, cup, h = content.text.split(', ')
            bust, cup, h = bust[-15:].replace(' / ', '-'), cup[-5], h[4:7] 
            if 'n/a' in bust or 'n/a' in cup or 'n/a' in h:continue 
            file_name = '-'.join([bust, cup, h]).replace(' (cm)', '').replace(': ', '').replace('?', '').replace('*', '').replace('/', '').replace('\\', '')
            
            if not os.path.exists(os.path.join("./images", "unprocessed")):
                os.mkdir(os.path.join("./images", "unprocessed"))  # 建立資料夾
            
            img = requests.get(link)  # 下載圖片
            if not os.path.exists(os.path.join("./images", "unprocessed") + '/' + file_name + "_" + str(index+1) + ".jpg"):
                with open(os.path.join("./images", "unprocessed") + '/' + file_name + "_" + str(index+1) + ".jpg", "wb") as file:  # 開啟資料夾及命名圖片檔
                    file.write(img.content)  # 寫入圖片的二進位碼

if __name__ == "__main__":
    get_image()