
import os,sys,shutil
'''
testing code for copying file to target path
'''
# C:\Users\JiangYufeng\Desktop\WLASL2000\WLASL2000
book_list = ['69241', '65225', '68011', '68208', '68012', '70212', '70266', '07085', '07086', '07087', '07069', '07088', '07089', '07090', '07091', '07092', '07093', '07068', '07094', '07095', '07096', '07097', '07070', '07098', '07099', '07071', '07072', '07073', '67424', '07074', '07075', '07076', '07077', '07078', '07079', '07080', '07081', '07082', '07083', '07084']
source_folder=r'C:\Users\JiangYufeng\Desktop\WLASL2000\WLASL2000'
new_path = r'C:\Users\JiangYufeng\Desktop\WLASL2000\Book'
num = 0
file_list=os.listdir(source_folder)
for file_obj in file_list:
    file_path=os.path.join(source_folder,file_obj)
    file_name,file_extend=os.path.splitext(file_obj)

    for i in range(len(book_list)):
        if file_name == book_list[i]:
            # new_name = str(num) + file_extend
            num += 1
            newfile_path = os.path.join(new_path, file_obj)
            shutil.copyfile(file_path,newfile_path)
        else:
            continue
    
