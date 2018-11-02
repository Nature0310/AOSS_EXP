## 实验说明


		cd <to random_experiment> # 执行路径
		# 执行命令参考
		python3 main_train_random.py --resume --range 1-5 --random-NF  --gpu 6  --savedir result_random/savedir
		# 参数说明
		--resume : 支持断点续，一直使用就可以
		--range 1-5: index 为1 到5 的5组实验，起止可以随意设置，总范围1-110,多个程序同时执行时最好range表示的范围不要交叉
		--random_NF: 关于结构的一些参数，一直使用就可以
		--gpu 6： 使用6号gpu
		--savedir: 结果存储路劲

		回收savedir里的文件

