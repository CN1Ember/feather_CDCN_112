import os
import shutil

filepath_ = '/home/lidaiyuan/feathernet2020/FeatherNet/submission/20210226_gan_2021-02-26_21:42:27_FaceFeatherNet_v3_0_submission_filename.txt'
predpath_ = '/home/lidaiyuan/feathernet2020/FeatherNet/submission/20210226_gan_2021-02-26_21:42:27_FaceFeatherNet_v3_0_submission.txt'
gtpath_ = '/home/lidaiyuan/feathernet2020/FeatherNet/submission/20210226_gan_2021-02-26_21:42:27_FaceFeatherNet_v3_0_submission_gt.txt'

transet_root = '/mnt/cephfs/dataset/face_anti_spoofing_lock/fas_nir_datasets/fas_dataset_nir_20200813_mask/trainset'
failure_case_path = '/mnt/cephfs/dataset/face_anti_spoofing_lock/fas_nir_datasets/failure_case/20210226_gan_2021-02-26_21:42:27'

print(filepath_)
f1 = open(filepath_,'r')
f2 = open(predpath_,'r')
f3 = open(gtpath_,'r')

filepthlist = f1.read().splitlines()
predlist = f2.read().splitlines()
gtlist = f3.read().splitlines()

filter_ = 'bmp_mask'
thr = 0.5
curr = 1e-5
err = 1e-5
tp = 1e-5
fp = 1e-5
tn = 1e-5
fn = 1e-5

for idx,filepath in list(enumerate(filepthlist)):
	# if filter_ in filepath:
		score = float(predlist[idx])
		label = gtlist[idx]
		if label == '1':
			if score > thr:
				tp = tp + 1
			else:
				fn = fn + 1
		else:
			if score < thr:
				tn = tn + 1
			else:
				fp = fp + 1			
		if score > thr and label == '1' or (score < thr) and label == '0':
			curr = curr + 1
		else:
			dstpath = filepath.replace(transet_root,failure_case_path)
			dstdirpth = os.path.dirname(dstpath)
			if not os.path.exists(dstdirpth):
				os.makedirs(dstdirpth)
			print(filepath,score,label)
			subfix = filepath.split('.')[-1]
			print(dstpath)
			shutil.copyfile(filepath,dstpath)
			err = err + 1
print(curr + err,'samples', curr,'prediction right','prediction accuracy:',curr / (curr + err))
print('tpr:',tp / (tp + fn),'fpr:', fp / (fp + tn),)
