# 733-Object Detection on Xray images
## CMPT-733 Programming for Big Data 2 (Simon Fraser University)
### Group members:Nattapat Juthaprachakul, Rui Wang, Siyu Wu, Yihan Lan

#### Object Detection in X-Ray Images
* The goal of this project is to use multiple algorithms, train multiple models, and report on comparative performance of each one. Performance of model is described by mean average precision scores(Object Detection metrics), also including accuracy and recall scores.

* The program is written in Python.


#### Requirements:
1. Tensorflow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection)
2. Tensorflow 1.15 (Tensorflow 2.0 does not work properly with Object Detection API)
3. NumPy version lower than 1.18 as it will have problem with COCO API
4. COCO API (for evaluation metric and all of our models used are pre-trained on COCO Dataset)
5. Google Cloud Platform for training models and evaluating results with new images (you can use Compute Engineer VM or Google Deeplearning VM)
6. Some other basic library for Data Science like scikin-learn, pandas, numpy, matplotlib, ..

#### Models:
* Download pre-trained models from Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

* Download Config files for each model (https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

#### In this project, we are using the following models:
1. SSD_mobilenet_v1
2. SSD_mobilenet_v1_fpn
3. SSD_inception_v2
4. SSD_resnet50
5. RFCN_resnet101
6. Faster_RCNN_resnet50
7. Faster_RCNN_resnet101
8. Faster_RCNN_inception_v2


#### Dataset:
* SIXray dataset consists of X-ray images from Beijing subway (https://github.com/MeioJane/SIXray)
##### Credit:
* (@INPROCEEDINGS{Miao2019SIXray,
    author = {Miao, Caijing and Xie, Lingxi and Wan, Fang and Su, chi and Liu, Hongye and Jiao, jianbin and Ye, Qixiang },
    title = {SIXray: A Large-scale Security Inspection X-ray Benchmark for Prohibited Item Discovery in Overlapping Images},
    booktitle = {CVPR},
    year = {2019} })

#### COPY from Local to Google Cloud Platform:

* single file: $ gcloud compute scp positive_train.record username@tensorflow-1-vm:. --zone us-west1-b

* folder: $ gcloud compute scp --recurse rfcn_resnet101 username@tensorflow-1-vm:./model/ --zone us-west1-b

#### COPY from Google Cloud Platform to Local:

* single file: $ gcloud compute scp username@tensorflow-1-vm:model.ckpt-200000 . --zone us-west1-b

* folder: $ gcloud compute scp --recurse username@tensorflow-1-vm:trained_rfcn_resnet101 . --zone us-west1-b

#### Training code: use Tensorflow Object Detection API (model/research/object_detection)

* $ python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/pipeline.config

#### Viewing Progress of Training via Tensorboard:

* $ gcloud compute firewall-rules create tensorboard-port --allow tcp:8008

* $ tensorboard --logdir=trained_model --port=8008

* Then look at the progress at http://[external-ip-of-Google-Cloud-VM]:6006


#### For starting Google Cloud VM in order to run Tensorflow Object Detection API
* Tensorflow Object Detection API (model/research)

* $ protoc object_detection/protos/*.proto --python_out=.

* $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

* $ python object_detection/builders/model_builder_test.py (testing whether it works or not)
