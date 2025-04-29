# Domain Knowledge Enhanced Vision-Language Pretrained Model for Dynamic Facial Expression Recognition (ACM MM2024)
![image](Framework.png)
This repo is the official PyTorch implementation for the paper "Domain Knowledge Enhanced Vision-Language Pretrained Model for Dynamic Facial Expression Recognition".

# Training
```
torchrun --nproc_per_node=2 --master_port=12348 main.py --config=configs/dfew/dfew.yaml
```
# Detailed Experimental Results
### DFEW
| Fold | Happy |Sad | Neutral | Angry | Surprise | Disgust | Fear | UAR | WAR |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 1    | 94.27       |76.77 | 78.41  |78.08    |69.07 |9.38 |	37.09 |	63.29 |	76.21|
| 2    | 95.95       |72.87 | 75.30  |70.56    |60.89 |24.05 |	41.46 |	63.01 |	73.22|
| 3    | 93.48       |72.59 | 79.05  |82.09    |64.59 |23.74 |	33.94 |	64.21 |	75.56|
| 4    | 93.44       |74.79 | 66.03 |84.10    |61.57 |30.53 |	52.71 |	66.16 |	74.48|
| 5    | 95.90       |83.93 | 76.50  |78.44    |61.93 |30.29 |	49.46 |	68.06 |	77.57|
| Avg  | 94.61       |76.19 | 75.06  |78.65    |63.61 |23.60 |	42.93 |	64.95 |	75.41|

The relevant run logs can be viewed log/DFEW.

### FERV39K
| Fold | Happy |Sad | Neutral | Angry | Surprise | Disgust | Fear | UAR | WAR |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 1    | 74.93      |60.94 | 60.31  |48.16    |22.98 |10.50 |	15.04 |	41.84 |	52.37|

The relevant run logs can be viewed log/FERV39K.
### MAFW
|Fold| Happy|	Sad|	Neutral	|Angry|	Surprise|	Disgust|	Fear	|contempt|	anxiety|	helplessness	|disappointment|	UAR|	WAR|
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|1   |  85.53 |	67.69 |	57.02 |	52.16 |	41.87 |	23.44 |	28.80 |	4.17 |	40.22 |	7.48 |	2.78 |	37.38 |	50.24 | 
|2   |  86.33 |	62.59 |	69.74 |	61.87 |	52.55 |	28.13 |	30.38 |	8.52 |	28.94 |	5.70 |	7.90 |	40.24 |	53.35 |
|3   |  87.10 |	68.37 |	67.55 |	62.95 |	54.21 |	44.53 |	72.01 |	6.34 |	39.89 |	3.85 |	2.78 |	46.32 |	59.36 |
|4   |  86.29 |	69.39 |	70.95 |	66.91 |	73.37 |	53.12 |	38.42 |	8.52 |	40.99 |	1.93 |	16.67 |	47.87 |	61.35 |
|5   |  89.92 |	85.03 |	58.59 |	65.47 |	50.00 |	28.35 |	31.15 |	6.52 |	50.79 |	9.62 |	0.00 |	43.22 |	58.49 |
|Avg |  87.03 |	70.61 |	64.77 |	61.87 |	54.40 |	35.51 |	40.15 |	6.81 |	40.17 |	5.72 |	6.03 |	43.01 |	56.56 |

The relevant run logs can be viewed log/MAFW.
# Acknowledgement
We appreciate the pioneer project [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP).
