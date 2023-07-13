from bs4 import BeautifulSoup
import requests
import os
import torch
from PIL import Image
from torchvision import transforms

trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
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

def delete_images():
    CLASS = ['big', 'small']
    types = ['train', 'val']
    tar_img = Image.open('images/train/big/01-W68-H101-G-165_43.jpg').convert("RGB")
    tar_tensor = trans(tar_img)
    for f_type in types:
        for img_type in CLASS:
            path = os.path.join('images', f_type, img_type)
            file_list = os.listdir(path)
            for file_name in file_list:
                path = os.path.join('images', f_type, img_type) + '/' + file_name
                if torch.equal(trans(Image.open(path).convert("RGB")), tar_tensor):
                    print('same')
                    os.remove(path)

if __name__ == "__main__":
    get_image()
    #delete_images()