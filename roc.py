import numpy as np

from scipy import interpolate

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def cal_metric(target, predicted,show = False, save = False):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    roc_result=open("./results/roc_0227_best.txt",'w')
    # roc_result.write("thresholds \t fpr \t tpr \n")
    for i in range(fpr.shape[0]):
        roc_result.write(str(thresholds[i])+ ' ' + str(fpr[i])+' '+str(tpr[i])+'\n')
    roc_result.close()
    _tpr = (tpr)
    _fpr = (fpr)
    tpr = tpr.reshape((tpr.shape[0],1))
    fpr = fpr.reshape((fpr.shape[0],1))
    scale = np.arange(0, 1, 0.00000001)
    function = interpolate.interp1d(_fpr, _tpr)
    y = function(scale)
    znew = abs(scale + y -1)
    eer = scale[np.argmin(znew)]
    FPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    TPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    for i, (key, value) in enumerate(FPRs.items()):
        index = np.argwhere(scale == value)
        score = y[index] 
        TPRs[key] = float(np.squeeze(score))
        print(score)
    auc = roc_auc_score(target, predicted)
    return scale,y
    # return ,eer,TPRs, auc,{'x':scale, 'y':y}
def save_result(scale,show = False,save = False):
    
    if show:
        plt.plot(scale, y)
        plt.show()
    if save:
        plt.plot(scale, y)
        plt.title('ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig('./results/feathernet_nir_0615.png')
    

if __name__ == "__main__":
   f1 = open('/mnt/cephfs/home/chenguo/code/feather_CDCN_112/submission/nir_20210612_2021-06-15_11:50:33_FaceFeatherNetA_NIR_0_submission_gt.txt','r')
   f2 = open('/mnt/cephfs/home/chenguo/code/feather_CDCN_112/submission/nir_20210612_2021-06-15_11:50:33_FaceFeatherNetA_NIR_0_submission.txt','r')
   label = [int(i) for i in f1.read().splitlines()]
   pre = [float(i) for i in f2.read().splitlines()]
   scale_nir , y1 = cal_metric(label,pre,False,True)

   f1 = open('/mnt/cephfs/home/chenguo/code/feather_CDCN_112/submission/nir_20210612_2021-06-15_11:58:29_CDCFeatherNetA_0_submission_gt.txt','r')
   f2 = open('/mnt/cephfs/home/chenguo/code/feather_CDCN_112/submission/nir_20210612_2021-06-15_11:58:29_CDCFeatherNetA_0_submission.txt','r')
   label = [int(i) for i in f1.read().splitlines()]
   pre = [float(i) for i in f2.read().splitlines()]
   scale_wcdc,y2 = cal_metric(label,pre,False,True)

   plt.plot(scale_nir, y1)
   plt.plot(scale_wcdc, y2)
   plt.title('ROC')
   plt.xlabel('FPR')
   plt.ylabel('TPR')
   plt.savefig('./results/feathernet_nir_0615.png')
