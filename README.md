# RED-SENet(introduction&show)

# 数据集(DATASETS)
在论文中，共使用4个数据集对RED-SENet去噪模型进行了验证，其中FH(胎儿心脏超声图像数据集)和GS（胆囊结石超声图像数据集）是实验室的私有数据集，HC18(HC18数据集官网：https://hc18.grand-challenge.org/)和CAMUS(CAMUS数据集官网：https://www.creatis.insa-lyon.fr/Challenge/camus/participation.html)是公开的超声图图像数据集。

# 使用(Use)
1.run python load.py to load the data.
2.run python main.py --mode='train' and --load_mode=0 to training . If the available memory(RAM) is more than 10GB, it is faster to run --load_mode=1.
3.run python main.py --mode='test' --test_iters=100000 to test.
# 结果(Result)
![image](https://user-images.githubusercontent.com/52170165/115249068-d786db00-a15a-11eb-9e78-ae10659c396d.png)

