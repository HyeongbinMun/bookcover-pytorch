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

3. 학습
```shell
python train.py
```

4. 인퍼런스
```shell
python inference.py
```
