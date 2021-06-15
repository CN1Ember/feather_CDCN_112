

filename='./data/nir_031802/exp_train_set_nir_21031802_exp_20210325205339_train_label.txt'

with open (filename,'r') as f:
    a=f.readlines()
    print(a[1])
    print("total label is ",len(a))
    num1=a.count(a[1])
    print("num of label {} is {}".format(a[1],a.count(a[1])))
    print(len(a)-num1)
    f.close()
print("read_ok")