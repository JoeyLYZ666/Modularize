import os
import sys
sys.path.append('..')
from itertools import combinations


if __name__ == '__main__':
    #数组的第一个元素
    model = ['simcnn', 'rescnn', 'incecnn'][0]
    #combinations(list , n):n代表从list中的元素选择出来n个的所有组合，这里是0-9每个数字作为一个元组
    #使用list将元组转换为元组列表（不使用list()会得到组合的迭代器对象，只能迭代获取里面的元素）
    #这里获得了cifar的所有类别
    #[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]——单元素元组需要包含一个逗号即使后面没有其他元素
    class_cifar_comb = list(combinations(list(range(10)), 1))
    class_svhn_comb = list(combinations(list(range(10)), 1))

    for class_cifar in class_cifar_comb:
        #输出到日志文件里面不能包含逗号，常用|管道符或者\t制表符
        #逗号常常作为CSV文件的数据分割符
        str_class_cifar = ''.join([str(i) for i in class_cifar])
        class_cifar = ','.join([str(i) for i in class_cifar])
        for class_svhn in class_svhn_comb:
            #[str(i) for i in class_svhn]是一个字符串列表会有如下形式
            #   0：'0' len(): 1 等其余属性
            #但是''.join([str(i) for i in class_svhn])是字符串，形式如下
            #   '0'
            str_class_svhn = ''.join([str(i) for i in class_svhn])
            #class_svhn是命令行参数，存在多个模块加入的时候需要逗号连接
            class_svhn = ','.join([str(i) for i in class_svhn])

            cmd = f'python -u src/experiments/reuse/reuse_modules.py ' \
                  f'--model {model} --class_cifar {class_cifar} --class_svhn {class_svhn}'
            print(cmd)
            os.system(cmd)
