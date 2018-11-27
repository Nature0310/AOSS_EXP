## 实验说明
### diverse 实验

	cd <to PNAS_search_idep> #执行路径
	#执行命令参考
	python3 PNA_main.py --diverse  --savedir search_result/diverse --dicdir model_acc/ --knum 64 --gpu 0 --distance 0.015 --id 0 


####参数说明

--diverse 指明表示利用diverse，保持使用即可。

--savedir 保存结果的路径。

--dicdir 存放已训练模型验证集准确率的位置 需要用个链接，连到公共存储区域共享。(目前git目录中该文件下存了一些已训练过模型的结果，可以拿来直接用加速)

--knum 每次选择多少个模型进行训练。目前固定64

--gpu 在指定序号gpu上运行程序

--distance 指定两个模型间的距离差值.(需要取不同值：0.0001, 0.0005, 0.0010, 0.0012, 0.0014，0.0016，0.0018，0.0020，0.0022，0.0025，0.0030)

--id 实验的编号，整型。（每个distance 跑5组实验，distance-0.0001 对应id：0，1，2，3，4；distance-0.0005对应id:5,6,7,8,9;distance-0.0010对应id:10,11,12,13,14;distance-0.0012对应id:15,16,17,18,19;distance-0.0014对应id:20,21,22,23,24;distance-0.0016对应id:25,26,27,28,29;distance-0.0018对应id:30,31,32,33,34;distance-0.0020对应id:35,36,37,38,39;distance-0.0022对应id:40,41,42,43,44;distance-0.0025对应id:45,46,47,48,49;distance-0.0030对应id:50,51,52,53,54）

* 实验回收savedir里的结果以及dicdir里训练过的模型结果
* 每一个distance 的5组重复性实验中，后启动的程序大概率会复用到之前实验跑过的模型结果，初步估计第一个程序会6天左右，后序复用结果会快一些。所以重复性实验可以通过延后启动的方式来缩短模型训练时间。

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
	--distance 指定两个模型间的距离差值，默认0.0015
	--id 实验的编号，重复5组，编号5，6，7，8，9。
