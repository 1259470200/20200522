import os
data_dir = r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\dataset\DRIVE\training'
for root, dirs, files in os.walk(data_dir):
    print('root', root)
    print('dirs', dirs)
    print('files', files)


def dataInfo(data_dir):  # 自定义函数用来获取数据信息，输入为数据所在路径，返回为一个元组(图像路径，标签路径)
    # 先读取所有的图像数据路径
    img_path = os.path.join(data_dir, 'images')
    imgs = os.listdir(img_path)
    imgs.sort(key=lambda x: int(x.split('_')[0]))  # 根据图片标号从小到大排序
    label_path = os.path.join(data_dir, '1st_manual')
    labels = os.listdir(label_path)
    labels.sort(key=lambda x: int(x.split('_')[0]))
    data_info = list()
    for i in range(len(imgs)):
        imgp = os.path.join(img_path, imgs[i])
        labelp = os.path.join(label_path, labels[i])
        data_info.append((imgp, labelp))
    return data_info
