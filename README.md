# DNAformer_simple

基础版的DNAformer深度学习纠错模型，基于论文Deep DNA Storage-Scalable and Robust DNA Storage via Coding Theory and Deep Learning

只是对论文中的模型的简单实现，DNA的数据是用python手动生成的，没有利用原文中的工具（无法找到他所提及的那篇参考文献）

具体模型效果如下：

模型参数：

实验参数

|参数名称| 数值| 备注|
|  ----  | ----  | ----  |
|N| 2|
|M| 1|
|C1| 0.1 |CSEloss 参数|
|C2| 100 |HMloss 参数
|样本数量| 1024|
|复制量| 500|
|碱基序列长度| 100|
|batch| 16|
|epoch| 10|
|Lr（学习速率）| 0.01|

受限于设备性能，所以很多参数不能与原文一致，抱歉！

模型概念图：

![image](https://user-images.githubusercontent.com/88192911/162429868-97164f9c-d5a2-4dc8-b97a-8700a24fe317.png)


模型测试结果：

![image](https://user-images.githubusercontent.com/88192911/159869769-e0c044f7-716d-448f-9a78-084a585c3f5a.png)

