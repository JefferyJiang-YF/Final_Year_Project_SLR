import json
import numpy as np
import os,sys,shutil
### Do classification on WLASL
# The output of this code is /WLASL/gloss*/video.mp4
#   gloss*: isolated sign language word 
new_dict = {}
with open("WLASL_v0.3.json","r") as f:
    load = json.load(f)
    # print("Done!")
    # print(load)
    #print(len(load)) # 2000
    #print(type(load)) # <class 'list'>
    # print(load[0]['instances'][0]["video_id"]) # list + dict + list + dict

# print(len(load[0]['instances'])) # 每个gloss视频的数量 ！！
Gloss = [] # Store the gloss to the list
for i in range(2000):
    Gloss.append(load[i]["gloss"])
    if load[i]["gloss"] == 'basketball':
        print(i)
print(Gloss[0:100])
defaultPath = r'C:\Users\JiangYufeng\Desktop\WLASL2000'
def find_diff_lowercase():
    actions = np.array(Gloss)
    AlphList = []
    for i in range(ord('a'), ord('z')+1):
        AlphList.append(chr(i))
    print(AlphList)
    # print(actions)
    # mkdirs for each gloss
    alist = []
    for i in range(len(Gloss)):
        for j in range(len(AlphList)):
            if Gloss[i] == AlphList[j]:
                alist.append(Gloss[i])
    print(alist)
    
def mkdir_():
    for action in actions:
        try:
            os.makedirs(os.path.join(defaultPath,action))
        except:
            pass
wlasl_100  = Gloss[0:100]
source_folder=r'C:\Users\JiangYufeng\Desktop\WLASL2000\WLASL2000'
# new_path = r'C:\Users\JiangYufeng\Desktop\WLASL2000\{}'.format()
def run():
    with open("video_id.json","r") as id_file:
        load_id = json.load(id_file)
        # print(type(load_id)) # <class 'dict'>
        
            
            # print(item) # output is each gloss
            # print(load_id[item]) # output is video_id for each gloss
            # print(len(load_id)) 
        file_list = os.listdir(source_folder)
        for file_obj in file_list:
            for item in load_id:
                new_path = r'C:\Users\JiangYufeng\Desktop\WLASL2000\{}'.format(item)
                file_path=os.path.join(source_folder,file_obj)
                file_name,file_extend=os.path.splitext(file_obj)
                for i in range(len(load_id[item])):
                    
                    # print(i,'current target copy path:', new_path,item)
                    if file_name == load_id[item][i]:
                        newfile_path = os.path.join(new_path, file_obj)
                        shutil.copyfile(file_path,newfile_path)
                        break
                    else:
                        if i == (len(load_id[item]) - 1 ):
                            break
                        else:
                            continue
