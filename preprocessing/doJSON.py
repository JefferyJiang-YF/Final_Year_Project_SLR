import json
import os
'''
testing code for json operation
'''
new_dict = {}
with open("WLASL_v0.3.json","r") as f:
    load = json.load(f)
    # print("Done!")
    # print(load)
    #print(len(load)) # 2000
    #print(type(load)) # <class 'list'>
    # print(load[0]['instances'][0]["video_id"]) # list + dict + list + dict

print(len(load[77]['instances'])) # 每个gloss视频的数量 ！！
Gloss = [] # Store the gloss to the list
for i in range(2000):
    Gloss.append(load[i]["gloss"])
    if load[i]["gloss"] == 'never':
        print(i)
    #  for j in range(load[i]['instances']):
#print(Gloss)


print(list(load[0]["instances"][0].values())[-1]) # 获取每个Gloss的id 如果要获取每个gloss视频id，修改第二个0的index作为循环索引遍历添加
# 例，获取gloss = book 的视频id
gloss_list = []
gloss_dict ={}
for a in range(18):
        
    print(list(load[77]["instances"][a].values())[-1])
for i in range(2000):
    for a in range (len(load[i]['instances'])):
        gloss_list.append(list(load[i]["instances"][a].values())[-1])
    #print(gloss_list)
    temp = {load[i]['gloss']:gloss_list}
    # print(temp)
    gloss_dict.update(temp)  
    gloss_list = []
    #print(gloss_dict)
# print(gloss_dict['book'])
json_str = json.dumps(gloss_dict)
with open("video_id.json","w") as json_file:
    json_file.write(json_str)