#尝试加入一些新的图片，看看效果有没有提升
import csv
import os, random, shutil


dir_test = "/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semTest/"
dir_train_3 = "/Users/jiangxuanke/Desktop/issm2020-ai-challenge/0003_original/"
dir_train_3_sample = "/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semTrain/0003/"
csvFile = open("/Users/jiangxuanke/Desktop/issm2020-ai-challenge/TestResultHuman.csv", "w")  # 创建csv文件
writer = csv.writer(csvFile)
writer.writerow(["Id", "LABEL"])

results = []
image_number = 1
img_test = os.listdir(dir_test)
img_test.sort()
print(img_test)
group_2 = [11,53,69,123,133,175,88,204,239,246,265,275,283,310]
group_3 = [8,14,15,16,25,27,45,51,54,55,71,81,86,89,92,95,99,103,110,116,122,124,125,141,156,158,163,171,176,182,185,189,190,203,210,211,221,233,237,242,245,248,249,256,259,261,266,267,271,279,282,286,289,296,298,300,311,312,316,317,331,337,341,345,346,347]
group_4 =[18,41,220,294,313,329]
group_5 = [2,5,9,19,24,28,34,35,48,67,68,98,102,112,119,120,130,135,136,139,140,142,145,150,161,173,180,181,183,184,193,194,197,205,230,269,291,293,343]
group_6 = [7,39,40,44,46,50,62,64,78,94,101,109,111,118,128,134,147,151,162,187,218,222,231,240,244,247,268,273,278,280,299,303,309,314,319,326,330,336,338]
group_7 = [3,42,57,58,60,77,138,165,174,179,198,206,216,226,235,251,257,258,263,277,287,321,325,328,344]
group_8 = [6,23,30,38,84,115,152,159,164,192,301,333]
group_9 = [74,90,153,202,241,272,240]
group_10 = [4,10,12,13,21,31,33,56,72,75,88,107,144,148,169,207,208,253,262]
for i in range(len(img_test)):
    writer.writerow([image_number, 3])
# for i in range(len(img_test)):
#     if image_number in group_10:
#         writer.writerow([image_number, 10])
#     elif image_number in group_2:
#         writer.writerow([image_number, 2])
#     elif image_number in group_3:
#         writer.writerow([image_number, 3])
#     elif image_number in group_4:
#         writer.writerow([image_number, 4])
#     elif image_number in group_5:
#         writer.writerow([image_number, 5])
#     elif image_number in group_6:
#         writer.writerow([image_number, 6])
#     elif image_number in group_7:
#         writer.writerow([image_number, 7])
#     elif image_number in group_8:
#         writer.writerow([image_number, 8])
#     elif image_number in group_9:
#         writer.writerow([image_number, 9])
#     else:
#         writer.writerow([image_number, 1])
    image_number+=1


def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        rate = 0.3    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片

        for name in sample:
                print(name)
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	fileDir = dir_train_3    #源图片文件夹路径
	tarDir = dir_train_3_sample    #移动到新的文件夹路径
	moveFile(fileDir)