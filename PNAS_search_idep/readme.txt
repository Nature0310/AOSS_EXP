�������̷����֣�
1��һ����topk��diversek�ֱ�ͨ���Լ����ɵ���������Ԥ������lstm������ѵ����
2��һ����topk��diversek���ɵ�������������һ���Ԥ��������ѵ����lstm������ѵ����
 ������һ�ֶ�Ӧ����PNA_main.py�� �ڶ��ֶ�Ӧ����PNA_main_dependent.py
 

cd <to PNAS_search_idep> #ִ��·��
#ִ������ο�
python3.6 PNA_main.py --diverse  --savedir search_result/diverse --dicdir model_acc/ --knum 256 --gpu 0 --distance 0.015
##����˵��1
--diverse ָ����ʾ����diverse�� ��д��ʾʹ��topk�������ϱ�֤���ַ���ʵ�����һ�¡�
--savedir ��������·����
--dicdir �����ѵ��ģ����֤��׼ȷ�ʵ�λ�� ��Ҫ�ø����ӣ����������洢������
--knum ÿ��ѡ����ٸ�ģ�ͽ���ѵ���� Ĭ��256��
--gpu ��ָ�����gpu�����г���
--distance ָ������ģ�ͼ�ľ����ֵ��Ĭ��0.015


python3.6 PNA_main_dependent.py  --savedir search_result_depend/result --dicdir model_acc/ --knum 256 --gpu 0 --distance 0.015
##����˵��2
--savedir ��������·����
--dicdir �����ѵ��ģ����֤��׼ȷ�ʵ�λ�� ��Ҫ�ø����ӣ����������洢������
--knum ÿ��ѡ����ٸ�ģ�ͽ���ѵ���� Ĭ��256��
--gpu ��ָ�����gpu�����г���
--distance ָ������ģ�ͼ�ľ����ֵ��Ĭ��0.015

