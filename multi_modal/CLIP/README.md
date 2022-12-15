# CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach

![CLIP](CLIP.png)



## Usage
train
```shell
python train.py \
--learning_rate \       # (default=1e-3)
--epoch \               # (default=10)
--batch_size \          # (default=256)
--train_images \        # train image directory 경로(default='/workspace/DATA/images/train')
--train_labels \        # train label directory 경로(default='/workspace/DATA/labels/train')
--val_images \          # val image directory 경로(default='/workspace/DATA/images/val')
--val_labels \          # val label directory 경로(default='/workspace/DATA/labels/val')
--save_path \           # model save path 경로(default='/workspace/result/')

```

inference
```shell
python eval.py \
--val_images \          # val image directory 경로(default='/workspace/DATA/images/val')
--val_labels \          # val label directory 경로(default='/workspace/DATA/labels/val')
--vocap \               # 해당 vocap 수(default=95)
--model_path \          # test model path(default='/workspace/result/last_model.pt')
```

