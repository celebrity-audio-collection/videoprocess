# Celebrity Audio Extraction

## Introduction

CN-Celeb, a large-scale Chinese celebrities dataset published by Center for Speech and Language Technology (CSLT) at Tsinghua University.

## Description

1. Collect audio data of 1,000 Chinese celebrities.

2. Automatically clip videos through a pipeline including face detection, face recognition, speaker validation and speaker diarization.

3. Create a benchmark database for speaker recognition community.

## Basic Methods

1. Environments: Tensorflow, PyTorch, Keras, MxNet

2. Face detection and tracking: RetinaFace and ArcFace models.

3. Active speaker verification: SyncNet model.

4. Speaker diarization: UIS-RNN model.

5. Double check by speaker recognition: VGG model.

6. Input: pictures and videos of POIs (Persons of Interest).

7. Output: well-labelled videos of POIs (Persons of Interest).

## Publication

```
@misc{fan2019cnceleb,
  title={CN-CELEB: a challenging Chinese speaker recognition dataset},
  author={Yue Fan and Jiawen Kang and Lantian Li and Kaicheng Li and Haolin Chen and Sitong Cheng and Pengyuan Zhang and Ziya Zhou and Yunqi Cai and Dong Wang},
  year={2019},
  eprint={1911.01799},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```

## Project Website

http://project.cslt.org

## OpenCV Tracker

https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

## RetinaFace

https://github.com/deepinsight/insightface/tree/master/RetinaFace

### References

```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}
```

## InsightFace

https://github.com/deepinsight/insightface

### Citation

```
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

## SyncNet

https://github.com/joonson/syncnet_python

### Citation

```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```

## UIS-RNN

https://github.com/google/uis-rnn

### Citation

```
@inproceedings{zhang2019fully,
  title={Fully supervised speaker diarization},
  author={Zhang, Aonan and Wang, Quan and Zhu, Zhenyao and Paisley, John and Wang, Chong},
  booktitle={International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6301--6305},
  year={2019},
  organization={IEEE}
}
```

## VGG-Speaker-Recognition

https://github.com/WeidiXie/VGG-Speaker-Recognition

### Citation

```
@InProceedings{Xie19,
  author       = "W. Xie, A. Nagrani, J. S. Chung, A. Zisserman",
  title        = "Utterance-level Aggregation For Speaker Recognition In The Wild.",
  booktitle    = "ICASSP, 2019",
  year         = "2019",
}
```

## VGG-Speaker-Recognition + UIS-RNN Implementation

https://github.com/taylorlu/Speaker-Diarization
