## 实验说明
### diverse 实验

	cd <to PNAS_search_idep> #执行路径
	#执行命令参考
	python3 PNA_main.py --diverse  --savedir search_result/diverse --dicdir model_acc/ --knum 64 --gpu 0 --distance 0.015 --id 0 


####参数说明

--diverse 指明表示利用diverse，保持使用即可。

--savedir 保存结果的路径。

<font color='red'>--dicdir 存放已训练模型验证集准确率的位置 需要用个链接，连到公共存储区域共享。</font>

--knum 每次选择多少个模型进行训练。

--gpu 在指定序号gpu上运行程序

--distance 指定两个模型间的距离差值.(需要取不同值：0.005, 0.015, 0.025, 0.05, 0.1)

--id 实验的编号，整型。（每个distance 跑5组实验，distance-0.005 对应id：0，1，2，3，4；distance-0.015对应id:5,6,7,8,9;distance-0.025对应id:10,11,12,13,14;distance-0.05对应id:15,16,17,18,19;distance-0.1对应id:20,21,22,23,24）

* 实验回收savedir里的结果以及dicdir里训练过的模型结果
* 每一个distance 的5组重复性实验中，后启动的程序大概率会复用到之前实验跑过的模型结果，初步估计第一个程序会10天左右，后序复用结果会快一些。所以重复性实验可以后启动来缩短模型训练时间。

### topk 实验

	cd <to PNAS_search_idep> #执行路径
	#执行命令参考
	python3 PNA_main.py --savedir search_result/top --dicdir model_acc/ --knum 64 --gpu 0 --id 0
	--savedir 保存结果的路径。
	<font color='red'>--dicdir 存放已训练模型验证集准确率的位置 需要用个链接，连到公共存储区域共享。</font>
	--knum 每次选择多少个模型进行训练。
	--gpu 在指定序号gpu上运行程序
	--id 实验编号: 需5组实验，编号0,1,2,3,4

### 实验3

	cd <to PNAS_search_idep> #执行路径
	python3 PNA_main_dependent.py  --savedir search_result_depend/result --dicdir model_acc/ --knum 64 --gpu 0 --distance 0.015 --id 0 
	##参数说明2
	--savedir 保存结果的路径。
	--dicdir 存放已训练模型验证集准确率的位置 需要用个链接，连到公共存储区域共享。
	--knum 每次选择多少个模型进行训练。
	--gpu 在指定序号gpu上运行程序
	--distance 指定两个模型间的距离差值，默认0.015
	--id 实验的编号，重复5组，编号5，6，7，8，9。
