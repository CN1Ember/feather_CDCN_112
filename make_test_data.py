import os
import shutil

dst = '/home/lidaiyuan/data/nir_test_data'
pos = ['positive','pos']
indoor = 'indoor'


file_src = '/home/lidaiyuan/FeatherNets_forNir/data/nir_adapt_0423/nir_val_add0605.txt'
label_src = '/home/lidaiyuan/FeatherNets_forNir/data/nir_adapt_0423/label_val_add0605.txt'

f = open(file_src,'r')
f2 = open(label_src,'r')

file_list = f.readlines()
label_list = f2.readlines()
count = 0
for i in range(len(file_list)):
    subfix = file_list[i].split('.')[-1].strip('\n')
    if label_list[i].strip('\n') == '1':
     shutil.copyfile(file_list[i].strip('\n'),os.path.join(dst,'pos','%d.%s'%(count,subfix)))
    else:
     shutil.copyfile(file_list[i].strip('\n'),os.path.join(dst,'neg','%d.%s'%(count,subfix)))
    count+=1