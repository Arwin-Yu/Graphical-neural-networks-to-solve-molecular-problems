# Graphical-neural-networks-to-solve-molecular-problems

# 一、Dataset
艾滋病数据集是由药物治疗计划（DTP）艾滋病抗病毒筛查引入的，该计划测试了40,000多个化合物抑制艾滋病复制的能力。筛选结果被评估并分为三类：

- 确认不活跃（CI）
- 确认活跃（CA）
- 确认中等活跃（CM）

我们进一步结合后两个标签，使其成为非活性（CI）和活性（CA和CM）之间的分类任务。


原始数据csv文件包含以下几列。

- "smiles"。分子结构的SMILES表示

- "活性"。筛选结果的三类标签。CI/CM/CA

- "HIV_active"。筛选结果的二进制标签。1（CA/CM）和0（CI）。

#  三、项目结构
## 3.1、config.py
config.py 存放的是模型搭建和模型训练时用到的超参数。同一个超参数可能有不同的设置，这是因为在模型训练阶段，使用了基于贝叶斯的超参数选择策略：mango。它可以在一个给定的参数空间中，搜索最优的参数组合。详见论文[1]


## 3.2、dataset.py
dataset.py是实现是继承自torch_geometric包中的Dataset类，其中分子的性质使用 rdkit 库中的 Chem 包的 Chem.MolFromSmiles 函数计算得到，其性质作为分子的特征将来送进图神经网络模型中做进一步计算。

## 3.3、dataset_featurer.py
dataset_featurer.py的功能与dataset.py一样，是dataset.py的简化版。具体来说，把dataset.py中获取分子特征的函数 _get_node_features， _get_edge_features 和 _get_adjacency_info 用deepchem库封装好的函数 feat.MolGraphConvFeaturizer 代替。

## 3.4、model.py
在图神经网络的模型构建中，特征提取阶段使用了基于attention机制的图卷积层[2]， Topk的池化层[3]和全局池化层。最后，将提取到的特征送入三层全连接层，输出得到预测结果。

## 3.5、oversampling_data.py
在 HIV 数据集中，无活性的分子结构数量要远远多于有活性的分子数量，简单说，这是一个不平衡的数据集。因此，oversampling_data.py 脚本的作用就是对有活性的分子进行重复采样，尽量减轻数据集不平衡带来的影响。

## 3.6、train.py
train.py 脚本中实现了模型的训练，验证和保存。在模型评估中，调用sklearn计算了Accuracy，Precision，Recall 和 F1-score。

## 3.7、utils.py
utils.py 脚本中实现了一些可能用到的功能，例如：从给定的 smiles 字符串中加载一个rdkit分子对象。将给定的 simles 字符串表示成图像等等

# 参考
[1] S. S. Sandha, M. Aggarwal, I. Fedorov and M. Srivastava, "Mango: A Python Library for Parallel Hyperparameter Tuning," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 3987-3991, doi: 10.1109/ICASSP40776.2020.9054609.

[2] Shi Y, Huang Z, Feng S, et al. Masked label prediction: Unified message passing model for semi-supervised classification[J]. arXiv preprint arXiv:2009.03509, 2020.

[3] Gao H, Ji S. Graph u-nets[C]//international conference on machine learning. PMLR, 2019: 2083-2092.

[4] https://github.com/deepfindr/gnn-project
