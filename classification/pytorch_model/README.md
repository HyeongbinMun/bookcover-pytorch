# Book cover Classifiaction

1. 패키지 설치
```shell
pip install -r requirements.txt
```

2. wandb api key 등록 및 확인
```shell
export WANDB_API_KEY=YOUR_API_KEY
echo $WANDB_API_KEY
```

3. wandb sweep을 이용한 하이퍼 파라미터 자동 탐색 (AutoML)
```shell
wandb sweep sweep.yaml
wandb agent <USERNAME/PROJECTNAME/SWEEPID>
```

4. train
```shell
python train.py
```

5. inference
```shell
python inference.py
```
