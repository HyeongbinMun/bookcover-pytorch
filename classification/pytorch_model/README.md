# Book cover BenchMark

1. docker 설정 및 접속 후 ssh 연결
```shell
docker-compose up -d
docker attach food-kt
passwd
service ssh start
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

4. 학습
```shell
python train.py
```

5. 인퍼런스
```shell
python inference.py
```