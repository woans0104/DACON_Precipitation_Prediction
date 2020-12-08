# SNUST_DACON_Precipitation_Prediction

__1.배경__

위성은 지구의 70%를 차지하고 있는 해양은 기상 정보는 필수입니다, 강수량 산출은 전지구의 물과 에너지 순환 이해에 중요합니다. 미항공우주국 NASA는 1970년대부터 위성을 이용 강수 산출 연구를 수행해 왔으며, 2014년부터 GPM (Global Precipitation Measurement) 미션을 통해 전 지구 강수량을 30분단위로 제공합니다. NASA에서 사용하고 있는 강수 산출 알고리즘은 물리기반의 베이지안 통계를 활용하고 있어, 전 영역에 대해 고품질 데이터베이스를 만드는 어려움이 있습니다. 인공지능을 활용 관측값 자체에서 어떠한 경험적 튜닝 없이 강수 산출을 시도하여 NASA보다 강력한 강수산출 인공지능 AI 에 도전!

__2.주최/주관/후원__

주최 : AI프렌즈, SIA, KAERI, DACON  
  
주관 : DACON  
  
후원 : 연구개발특구진흥재단, 셀렉트스타  

__3.참가자 대상__

일반인, 학생, 기업 등 누구나

- https://dacon.io/competitions/official/235591/overview/

### Measures
- mae(mean absolute error)
- f1-score

### Data Description 

- GPM(Global Precipitation Measurement) Core 위성의 GMI/DPR 센서에서 북서태평양영역 (육지와 바다를 모두 포함) 에서 관측된 자료
- 특정 orbit에서 기록된 자료를 40 X 40 형태로 분할(subset) 하여 제공
- pdf자료: https://dacon.io/competitions/official/235591/talkboard/400589
- 영상자료: https://dacon.io/competitions/official/235591/talkboard/400598

- train.zip
  - 2016~2018 년 관측된 자료 (76,345개)
  - 2016년 자료: orbit 번호 010462 ~ 016152 (25,653개)
  - 2017년 자료: orbit 번호 016154 ~ 021828 (25,197개)
  - 2018년 자료: orbit 번호 021835 ~ 027509 (25,495개)

- test.zip 
  - 2019년 관측된 자료 (2,416개)


### Logger
- Train Logger       : epoch, loss, mae/f1-score
- Test Logger        : epoch, loss, mae/f1-score

## Getting Started
### Requirements
- Python3 (3.6.8)
- PyTorch (1.2)
- torchvision (0.4)
- NumPy
- pandas
- matplotlib


### Baseline Train scripts
``` 
CUDA_VISIBLE_DEVICES=3 python3 main.py --loss-function mae --exp Dacon_exp/Baseline_unet_bach512_sch64--optim adam --initial-lr 0.0001 --batch-size 512 --arch unet --start-channel 64
```
### Train full scripts 
```
CUDA_VISIBLE_DEVICES=3 python3 main.py  \
--loss-function mae \
--exp Dacon_exp/Baseline_unet_bach512_sch64 \
--optim adam \
--initial-lr 0.0001 \
--batch-size 512 \
--arch unet \
--start-channel 64 \
```

| Args 	| Options 	| Description 	|
|:---------|:--------|:----------------------------------------------------|
| trn-root 	|  [str] 	| dataset locations. 	|
| tst-root | [str] | dataset locations. |
| arch 	| [str] | model architecture.  default : unet |
| start-channel 	| [int] | First feature map size.  default : 32	 |
| batch-size 	| [int] 	| number of samples per batch. default : 8  |
| epochs 	| [int] 	| number of epochs for training. default : 200  |
| lr-schedule 	| [int]	| epoch decay 0.1. 	defalut : [20,30,35]  |
| learning_rate 	| [float] 	| learning rate. defalut : 0.1	|
| weight-decay 	| [float]	| weight-decay. 	defalut : 0.0005|
| loss-function 	| [str]	| loss function  defalut : mae  |
| optim 	| [str]	| optimizer  defalut : sgd  |
| exp 	| [str] 	| save folder name.  |


### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (2015-Ronneberrger et al.) \
[2] RainNet v1. 0: a convolutional neural network for radar-based precipitation nowcasting. (2020-Ayzel et al.) \
[3] PERSIANN-CNN: Precipitation Estimation from Remotely Sensed Information Using Artificial Neural Networks–Convolutional Neural Networks.(2019-Sadeghi et al.) \
[4] cGANs : Conditional Generative Adversarial Networks for Near Real-Time Precipitation Estimation from Multispectral GOES-16 Satellite Imageries—PERSIANN-cGAN. (2019-Hayatbini et al.)






