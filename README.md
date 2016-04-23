We provide evaluation code of the PRW dataset in this repository. Please kindly cite the Arxiv paper if you use this dataset.

Liang Zheng*, Hengheng Zhang*, Shaoyan Sun*, Manmohan Chandraker, Qi Tian, "Person Re-identification in the Wild", arXiv:1604.02531, 2016. (* equal contribution)

This code implements the baseline using DPM as detector and BoW+XQDA as recognizer.
With the prepared codes, you can obtain some baseline results with 3 steps.

1. Download the PRW dataset, and unzip it in the folder "PRW"

2. Run "metric_learning.m" to train the recognition model. 

3. Run "baseline_dpm_bow.m" to test the PRW dataset. 

Note that the DPM detection results are provided in folder "data". In future release, we will provide detection codes and more recognition codes.

Please let me know if you have any problems. liangzheng06@gmail.com
