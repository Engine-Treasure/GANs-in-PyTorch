# Conditional GANs implemented in PyTorch

The paper: [A Study of Smoothing Methods for Language Models Applied to Information Retrieval](https://arxiv.org/pdf/1411.1784.pdf)

This work tries to reproduce cGAN (Conditional GAN) refer only to the paper, without any later tricks but an alternative heuristic loss function. See [model1.py](model.py)

| Discrinator | Generator |
| --- | --- |
| X Input ∈ R^784; y input ∈ $r^10$ | z Input ∈ $r^100$; y input ∈ $r^10$ |
| 240,5 Maxout; OneHot. 50,5 Maxout. | FC. 1000 RELU; OneHot. FC. 200 RELU|
| Dropout(0.5). 290,4 Maxout. | Dropout(0.5). FC. 784 RELU. Sigmoid |
| Dropout(0.5). FC. Sigmoid. | |


| Hype-parameters | learning rate | decay facotr |momentum | optimizer |
| --- | --- | --- | --- | --- |
| Values | 0.1->0.000001 | 1.00004 | >0.7 | SGD |

[model2.py](model2.py) is used as a contrast model which has additional Batch Normlization layer in Generator.

| Discrinator | Generator |
| --- | --- |
| X Input ∈ R^784; y input ∈ $r^10$ | z Input ∈ $r^100$; y input ∈ $r^10$ |
| 240,5 Maxout; OneHot. 50,5 Maxout. | FC. BatchNorm. 1000 RELU; OneHot. FC. BatchNorm. 200 RELU |
| Dropout(0.5). 290,4 Maxout. | Dropout(0.5). FC. 784 RELU. Sigmoid |
| Dropout(0.5). FC. Sigmoid. | |

## Result

| Name | epoch1 | epoch10 | epoch50 | gif | remarks |
| --- | --- | --- | --- | --- | --- |
| Model1 | ![model1_epoch_1_iteration_500](samples/fake/fake_sample_0_500.png) | ![model1_epoch_10_iteration_500](samples/fake/fake_sample_10_300.png) | ![epoch_50_iteration_500](samples/fake/fake_sample_50_500.png) | ![fake.gif](samples/fake/fake.gif) | It still need time to converge. |
| Model2 | ![model2_epoch_1_iteration_500](samples2/fake/fake_sample_0_500.png) | ![model2_epoch_10_iteration_500](samples2/fake/fake_sample_10_400.png) | ![epoch_50_iteration_400](samples2/fake/fake_sample_50_400.png) | ![fake.gif](samples2/fake/fake.gif) | It shows that BN really herlps accelerating training. |


