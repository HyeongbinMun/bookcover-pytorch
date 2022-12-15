# Book cover Classifiaction

1. wandb api key 등록 및 확인
```shell
export WANDB_API_KEY=YOUR_API_KEY
echo $WANDB_API_KEY
```

2. wandb sweep을 이용한 하이퍼 파라미터 자동 탐색 (AutoML)
```shell
wandb sweep sweep.yaml
wandb agent <USERNAME/PROJECTNAME/SWEEPID>
```

3. train
```shell
python train.py \
--data_dir \    # 데이터셋 경로(default='/hdd/book-covers-split/')
--save_dir \    # 저장 경로(default='/hdd/model/book')
--model \       # model 선택(default='tf_efficientnet_b4')
--img_size \    # 이미지 사이즈(default=224)
--epochs \      # 반복횟수(default=200)
--batch_size \  # 배치 사이즈(default=32)

```

4. inference
```shell
python inference.py \
--data_dir \    # 데이터셋 경로(default='/hdd/book-covers-split/')
--checkpoint \  # 체크포인트 모델 경로(default='/hdd/sy/weights/book/ckpt_best.pt')
--model \       # model 선택(default='tf_efficientnet_b4')
--batch_size \  # 배치 사이즈(default=128)
--img_size \    # 이미지 사이즈(default=384)
--save_dir \    # 저장 경로(default='/hdd/model/book')
--csv_name \    # 저장되는 csv 이름(default='test')
```
