import easyocr

# 添加 gpu=True 参数
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
result = reader.readtext('1.jpg')
print(result)