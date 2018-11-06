搜索过程分两种：
1）一种是topk和diversek分别通过自己生成的样本来对预测器（lstm）进行训练。
2）一种是topk和diversek生成的样本整合起来一起对预测器进行训练（lstm）进行训练。
 上述第一种对应的是PNA_main.py， 第二种对应的是PNA_main_dependent.py
 

cd <to PNAS_search_idep> #执行路径
#执行命令参考
python3.6 PNA_main.py --diverse  --savedir search_result/diverse --dicdir model_acc/ --knum 256 --gpu 0 --distance 0.015
##参数说明1
--diverse 指明表示利用diverse， 不写表示使用topk。基本上保证两种方法实验次数一致。
--savedir 保存结果的路径。
--dicdir 存放已训练模型验证集准确率的位置 需要用个链接，连到公共存储区域共享。
--knum 每次选择多少个模型进行训练。 默认256个
--gpu 在指定序号gpu上运行程序
--distance 指定两个模型间的距离差值，默认0.015


python3.6 PNA_main_dependent.py  --savedir search_result_depend/result --dicdir model_acc/ --knum 256 --gpu 0 --distance 0.015
##参数说明2
--savedir 保存结果的路径。
--dicdir 存放已训练模型验证集准确率的位置 需要用个链接，连到公共存储区域共享。
--knum 每次选择多少个模型进行训练。 默认256个
--gpu 在指定序号gpu上运行程序
--distance 指定两个模型间的距离差值，默认0.015

