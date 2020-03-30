Protocol Buffers - Google's data interchange format
Copyright 2008 Google Inc.
https://developers.google.com/protocol-buffers/

This package contains a precompiled binary version of the protocol buffer
compiler (protoc). This binary is intended for users who want to use Protocol
Buffers in languages other than C++ but do not want to compile protoc
themselves. To install, simply place this binary somewhere in your PATH.

If you intend to use the included well known types then don't forget to
copy the contents of the 'include' directory somewhere as well, for example
into '/usr/local/include/'.

Please refer to our official github site for more installation instructions:
  https://github.com/protocolbuffers/protobuf

COPY local to cloud:
gcloud compute scp positive_train.record tensorflow-2-vm:.
gcloud compute scp --recurse rfcn_resnet101 njuthapr@tensorflow-1-vm:./model/ --zone us-west1-b
gcloud compute scp --recurse faster_rcnn_resnet101 njuthapr@tensorflow-1-vm:./model/ --zone us-west1-b
gcloud compute scp --recurse inference_graph njuthapr@tensorflow-1-vm:./trained_rfcn_resnet101/ --zone us-west1-b
gcloud compute scp cloud_run_infer_.py njuthapr@tensorflow-1-vm:./models/research/object_detection/ --zone us-west1-b


gcloud compute scp --recurse positive_test njuthapr@tensorflow-1-vm:./new_dataset/ --zone us-west1-b
gcloud compute scp --recurse negative_train njuthapr@tensorflow-1-vm:./new_dataset/ --zone us-west1-b
gcloud compute scp --recurse negative_test njuthapr@tensorflow-1-vm:./new_dataset/ --zone us-west1-b
gcloud compute scp label_test_train.csv njuthapr@tensorflow-2-vm:. --zone us-west1-b
gcloud compute scp --recurse negative_img11 njuthapr@tensorflow-1-vm:./new_dataset/ --zone us-west1-b
gcloud compute scp --recurse negative_img12 njuthapr@tensorflow-1-vm:./new_dataset/ --zone us-west1-b
gcloud compute scp --recurse negative_img11 njuthapr@tensorflow-2-vm:./new_dataset/ --zone us-west1-b
gcloud compute scp --recurse negative_img12 njuthapr@tensorflow-2-vm:./new_dataset/ --zone us-west1-b


COPY cloud to local:
gcloud compute scp thenatzzz@tensorflow-2-vm:./xray_models/ssd_inceptionv2/model.ckpt-200000* . --zone us-west1-b
gcloud compute scp --recurse njuthapr@tensorflow-1-vm:trained_rfcn_resnet101 . --zone us-west1-b
gcloud compute scp --recurse njuthapr@tensorflow-1-vm:trained_ssd_mobilenet_fpn . --zone us-west1-b
gcloud compute scp  --recurse njuthapr@tensorflow-1-vm:trained_rfcn_resnet101 . --zone us-west1-b

gcloud compute scp  --recurse njuthapr@tensorflow-2-vm:trained_faster_rcnn_resnet50 . --zone us-west1-b
gcloud compute scp  njuthapr@tensorflow-1-vm:label_score_rfcn_resnet101.csv . --zone us-west1-b



thenatzzz@DESKTOP-LJRLU7E MINGW64 /d/Coding/SFU_CA/CMPT-733/groupproject/models/research/object_detection
$ python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/pipeline.config
python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/pipeline.config

Tensorboard: (https://stackoverflow.com/questions/33836728/view-tensorboard-on-docker-on-google-cloud)
thenatzzz@tensorflow-2-vm:~/xray_models$ gcloud compute firewall-rules create tensorboard-port --allow tcp:6006
gcloud compute firewall-rules create tensorboard-port --allow tcp:8008
thenatzzz@tensorflow-2-vm:~/xray_models$ tensorboard --logdir=test2 --port=6006
tensorboard --logdir=trained_faster_rcnn_resnet50 --port=8008
http://35.197.75.253:6006/#images

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
