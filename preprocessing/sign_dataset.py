import json
import math
import os
import random
import cv2
import numpy as np
import os,sys,shutil
'''
1. 处理的目标文件格式为 file_name.npy
2. 处理目标文件具备以下特征
    a.长度不一样
'''

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)           

# def mkdir_actions(actions):
#     DATA_PATH = os.path.join(r'C:\deep-learning\HKMU\fyp_LSTM\ActionDetectionforSignLanguage\MP_KeyPointData\new-test') 
#     for action in actions:
#         for root, dirs, files in os.walk(r"C:\deep-learning\HKMU\fyp_LSTM\ActionDetectionforSignLanguage\data\test\{}".format(action)):
#             for sequence in range (len(files)):
#                 try:
#                     os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#                 except:
#                     pass

import os
import numpy as np

def mkdir_actions(base_path, data_path, actions):
    for action in actions:
        action_source_folder = os.path.join(base_path, action)
        if os.path.exists(action_source_folder):
            files = os.listdir(action_source_folder)
            for sequence in range(len(files)):
                try:
                    os.makedirs(os.path.join(data_path, action, str(sequence)))
                except FileExistsError:
                    pass
        else:
            print(f"Folder not found: {action_source_folder}")


def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample

def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1
    frames_to_sample = []
    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))

    return frames_to_sample

def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []
    if num_frames > num_samples:
        frames_skip = set()

        num_skips = num_frames - num_samples
        interval = num_frames // num_skips

        for i in range(frame_start, frame_end + 1):
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)

        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i)
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample

def padding_smaller(frame_start, frame_end, num_samples, source_folder):
    num_frames = frame_end - frame_start + 1

    if num_frames < num_samples:
        num_difference = num_samples - num_frames
        for i in range(num_difference):
            temp_np = np.zeros(258) # 1662/258 个关键点坐标
            np.save(os.path.join(source_folder ,str(frame_end+1)),temp_np)
            frame_end += 1
    return [ i for i in range(30)] #只需要30帧视频图片

def normalize_video(source_folder,new_path):
    leng = len(os.listdir(source_folder)) # 文件夹中文件总数
    for i in range(leng):

        process_path = os.path.join(source_folder + '{}'.format(i))
        pre_process_path = os.path.join(new_path + '{}'.format(i))
        file_list = os.listdir(process_path)
        frame_start = 0
        frame_end = len(file_list) - 1
       
        num = 0
        # if len(file_list) < 10:
        #     no_frame = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, 5, num_copies=6)
        # elif len(file_list) < 15:
        #     no_frame = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, 10, num_copies=3)
        # elif len(file_list) < 30:
        #     no_frame = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, 15, num_copies=2)
        if len(file_list) < 30:
            no_frame = padding_smaller(frame_start, frame_end, 30, pre_process_path)
        elif len(file_list)>30 and len(file_list)<=60:
            no_frame = rand_start_sampling(frame_start, frame_end, 30)
        elif len(file_list)> 60:
            no_frame = sequential_sampling(frame_start, frame_end, 31)
        else:
            no_frame = [ i for i in range(30)]

        for file_obj in file_list:
            file_path=os.path.join(process_path,file_obj)
            file_name,file_extend=os.path.splitext(file_obj)
                    
            for j in range(len(no_frame)):
                if file_name == str(no_frame[j]):
                    new_name = str(num) + file_extend
                    num += 1
                    newfile_path = os.path.join(pre_process_path, new_name)
                    shutil.copyfile(file_path,newfile_path)
                else:
                    continue
            
if __name__ == '__main__':

    """WLASL-10"""
    # actions = np.array(['book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy', 'cousin'])
    # # mkdir_actions(actions)
    # for action in actions:
    #     source_folder = r'C:/deep-learning/HKMU/fyp_LSTM/ActionDetectionforSignLanguage/MP_KeyPointData/train/{}/'.format(action)
    #     new_path = r'C:/deep-learning/HKMU/fyp_LSTM\ActionDetectionforSignLanguage/MP_KeyPointData/new-train/{}/'.format(action)
    #     normalize_video(source_folder, new_path)
    """ 
    This code will process train, validation, and test data, creating the respective normalized directories. 
     Make sure to use the correct paths according to your dataset organization.
     """
    wlasl_40 = ['deaf', 'fine', 'help', 'no', 'thin', 'walk', 'year', 'yes', 'all', 'black', 'cool', 'finish', 'hot', 'like', 'many', 'mother', 'now', 'orange', 'table', 'thanksgiving', 'what', 'woman', 'bed', 'blue', 'bowling', 'can', 'dog', 'white', 'wrong', 'accident', 'apple', 'bird', 'change', 'color', 'corn', 'cow', 'dance', 'dark', 'doctor']
    actions = np.array(wlasl_40)
    
    data_types = ['train', 'val', 'test']

    for data_type in data_types:
        base_path = f"keypoints\{data_type}"
        data_path = f"normalized\{data_type}"
        mkdir_actions(base_path,data_path, actions)

        for action in actions:
            action_source_folder = os.path.join(base_path, action,'')
            action_new_path = os.path.join(data_path, action,'')
            normalize_video(action_source_folder, action_new_path)
