Dataset: https://github.com/MeioJane/SIXray
(@INPROCEEDINGS{Miao2019SIXray,
    author = {Miao, Caijing and Xie, Lingxi and Wan, Fang and Su, chi and Liu, Hongye and Jiao, jianbin and Ye, Qixiang },
    title = {SIXray: A Large-scale Security Inspection X-ray Benchmark for Prohibited Item Discovery in Overlapping Images},
    booktitle = {CVPR},
    year = {2019} })



COPY from Local to Google Cloud Platform:
single file:$ gcloud compute scp positive_train.record username@tensorflow-1-vm:. --zone us-west1-b
folder:     $ gcloud compute scp --recurse rfcn_resnet101 username@tensorflow-1-vm:./model/ --zone us-west1-b

COPY from Google Cloud Platform to Local:
single file:$ gcloud compute scp username@tensorflow-1-vm:./xray_models/ssd_inceptionv2/model.ckpt-200000 . --zone us-west1-b
folder:     $ gcloud compute scp --recurse username@tensorflow-1-vm:trained_rfcn_resnet101 . --zone us-west1-b


Training code: use Tensorflow Object Detection API (model/research/object_detection)
$ python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/pipeline.config

Viewing Progress of Training via Tensorboard:
$ gcloud compute firewall-rules create tensorboard-port --allow tcp:8008
$ tensorboard --logdir=trained_model --port=8008
Then look at the progress at http://<external-ip-of-Google-Cloud-VM>:6006

python model_main.py --logtostderr --model_dir=/home/thenatzzz/xray_models/test/ --pipeline_config_path=/home/thenatzzz/xray_models/config_pipeline/pipeline.config
tensorboard --logdir=training/ex3 --samples_per_plugin=images=30

For starting VM: run
thenatzzz@tensorflow-2-vm:~/models/research$
protoc object_detection/protos/*.proto --python_out=.
thenatzzz@tensorflow-2-vm:~/models/research$
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
thenatzzz@tensorflow-2-vm:~/models/research$
python object_detection/builders/model_builder_test.py

thenatzzz@tensorflow-2-vm:~/models/research/object_detection$ python model_main.py --logtostderr --model_dir=/home/thenatzzz/xray_models/test/ --pipeline_config_path=/home/thenatzzz/xray_models/config_pipeline/pipeline.config
thenatzzz@tensorflow-2-vm:~/models/research/object_detection$
python model_main.py --logtostderr --model_dir=/home/thenatzzz/xray_models/ssd_resnet50_v1/ --pipeline_config_path=/home/thenatzzz/xray_models/config_model/ssd_resnet50_v1/pipeline.config
python model_main.py --logtostderr --model_dir=/home/njuthapr/trained_ssd_mobilenet_fpn/ --pipeline_config_path=/home/njuthapr/model/ssd_mobilenet_fpn/pipeline.config
python model_main.py --logtostderr --model_dir=/home/njuthapr/trained_faster_rcnn_resnet50/ --pipeline_config_path=/home/njuthapr/model/faster_rcnn_resnet50/pipeline.config
python model_main.py --logtostderr --model_dir=/home/njuthapr/trained_faster_rcnn_resnet101/ --pipeline_config_path=/home/njuthapr/model/faster_rcnn_resnet101/pipeline.config
