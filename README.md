# Domain Knowledge Enhanced Vision-Language Pretrained Model for Dynamic Facial Expression Recognition (ACM MM2024)
![image](Framework.png)
This repo is the official PyTorch implementation for the paper "Domain Knowledge Enhanced Vision-Language Pretrained Model for Dynamic Facial Expression Recognition".

# Training
```
torchrun --nproc_per_node=2 --master_port=12348 main.py --config=configs/dfew/dfew.yaml
```
