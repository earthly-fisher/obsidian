1、做一个特征转化的层，固定层，不学习的那种，找一下resnet50中每一层针对的特征偏向。
2、每一层layer之间建一个结构，对resnet与新增的结构施加不同的学习率与优化器进行处理，换句话说就是参数分组