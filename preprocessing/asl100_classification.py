import json
import numpy as np
import os,sys,shutil
asl_100 = ['book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy', 'cousin', 'deaf', 'fine', 'help', 'no', 'thin', 'walk', 'year', 'yes', 'all', 'black', 'cool', 'finish', 'hot', 'like', 'many', 'mother', 'now', 'orange', 'table', 'thanksgiving', 'what', 'woman', 'bed', 'blue', 'bowling', 'can', 'dog', 'family', 'fish', 'graduate', 'hat', 'hearing', 'kiss', 'language', 'later', 'man', 'shirt', 'study', 'tall', 'white', 'wrong', 'accident', 'apple', 'bird', 'change', 'color', 'corn', 'cow', 'dance', 'dark', 'doctor', 'eat', 'enjoy', 'forget', 'give', 'last', 'meet', 'pink', 'pizza', 'play', 'school', 'secretary', 'short', 'time', 'want', 'work', 'africa', 'basketball', 'birthday', 'brown', 'but', 'cheat', 'city', 'cook', 'decide', 'full', 'how', 'jacket', 'letter', 'medicine', 'need', 'paint', 'paper', 'pull', 'purple', 'right', 'same', 'son', 'tell', 'thursday']
asl_dict = {}

def mkdir():
    a_path = r'C:\deep-learning\HKMU\fyp_LSTM\ActionDetectionforSignLanguage\data\test'
    actions = np.array(asl_100)
    for action in actions:
        try:
            os.makedirs(os.path.join(a_path,action))
        except:
            pass

def addIndex():
    for i in range(len(asl_100)):
        temp = {str(i):asl_100[i]} 
        asl_dict.update(temp)

addIndex()
source_path = r'C:\Users\JiangYufeng\Desktop\gloss'
def run():
    with open('nslt_100.json',"r") as f:


        load = json.load(f)

        # print(type(load)) # <class 'dict'>
        # print(len(load)) 2038

        for key_L, value_L in load.items():
            for key_D, value_D in asl_dict.items():

            #print(value.get("subset")) 获取其分类
            #print(value.get("action")[0]) 获取视频索引
            #print(asl_dict.get("1"))
                if value_L.get("subset") == 'train':
                    new_source_path = r'C:\deep-learning\HKMU\fyp_LSTM\ActionDetectionforSignLanguage\data\train'
                    if str(value_L.get("action")[0]) == key_D:
                        #"train"
                        search_path = os.path.join(r'C:\Users\JiangYufeng\Desktop\gloss',value_D)
                        new_path = os.path.join(new_source_path,value_D)
                        file_list=os.listdir(search_path)

                        for file_obj in file_list:

                            file_path=os.path.join(search_path,file_obj)
                            file_name,file_extend=os.path.splitext(file_obj)
                            if file_name == str(key_L):
                                newfile_path = os.path.join(new_path, file_obj)
                                shutil.copyfile(file_path,newfile_path)

                elif value_L.get("subset") == 'val':
                    new_source_path = r'C:\deep-learning\HKMU\fyp_LSTM\ActionDetectionforSignLanguage\data\val'
                    if str(value_L.get("action")[0]) == key_D:
                        #"val"
                        search_path = os.path.join(r'C:\Users\JiangYufeng\Desktop\gloss',value_D)
                        new_path = os.path.join(new_source_path,value_D)
                        file_list=os.listdir(search_path)

                        for file_obj in file_list:

                            file_path=os.path.join(search_path,file_obj)
                            file_name,file_extend=os.path.splitext(file_obj)
                            if file_name == str(key_L):
                                newfile_path = os.path.join(new_path, file_obj)
                                shutil.copyfile(file_path,newfile_path)

                else:
                    new_source_path = r'C:\deep-learning\HKMU\fyp_LSTM\ActionDetectionforSignLanguage\data\test'
                    if str(value_L.get("action")[0]) == key_D:
                        #"test"
                        search_path = os.path.join(r'C:\Users\JiangYufeng\Desktop\gloss',value_D)
                        new_path = os.path.join(new_source_path,value_D)
                        file_list=os.listdir(search_path)

                        for file_obj in file_list:

                            file_path=os.path.join(search_path,file_obj)
                            file_name,file_extend=os.path.splitext(file_obj)
                            if file_name == str(key_L):
                                newfile_path = os.path.join(new_path, file_obj)
                                shutil.copyfile(file_path,newfile_path)
run()
#print(os.path.join('C:\deep-learning\WLASL\data','test'))