## 環境準備
python: 3.7.9  
pip install opencv-contrib-python==3.4.2.17  
pip install Matplotlib==3.1.1  
pip install pyqt5==5.15.1  
pip install pyqt5-tools  
pip install pyinstaller  

python 路徑: C:\Users\shaojason999\AppData\Local\Programs\Python\Python37\Scripts\  
pyqt5 路徑: C:\Users\shaojason999\AppData\Local\Programs\Python\Python37\Lib\site-packages\pyqt5_tools\Qt\bin\designer.exe  

## 程式碼操作
1. 程式碼修改:
	(1) 介面使用 pyqt5 設計 .ui
	(2) 轉成 .py: pyuic5 -o hw1_1.py hw1_1.ui
	(3) 主程式寫在 main.py，然後 import hw1_1.py，如此避免介面更改後主程式碼被蓋掉
2. 編譯:
	$python main.py
3. 產生執行檔
	$pyinstaller -F main.py
	產生 dist/main.exe，記得要移動 main.exe 位置，因為要讀取圖片檔，否則會出現錯誤 Failed to execute script(https://blog.csdn.net/chouzhou9701/article/details/88850496)

## 注意
1. ui 設計時，最外層的物件不需要 layout，裡面用 layout 就好(有跳出錯誤可能可以不用理)
