# Book cover BenchMark

현 문서는 book cover classification과 text-recognition을 통한 text data 생성 및 multi modal 성능 평가를 진행하기 위한 benchmark dataset이다.

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
