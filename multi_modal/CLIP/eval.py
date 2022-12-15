import argparse
import torch
import clip
import random
import numpy as np

from PIL import Image
from train import dataloader
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_images', type=str, default='/workspace/DATA/images/val')
    parser.add_argument('--val_labels', type=str, default='/workspace/DATA/labels/val')
    parser.add_argument('--vocap', type=int, default=95)
    parser.add_argument('--model_path', type=str, default='/workspace/result/ver1/best_model.pt')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    d_test = dataloader(args.val_images, args.val_labels)

    model.load_state_dict(torch.load(args.model_path))
    NUM_NEG = args.vocap
    NUM_TEST = 1000

    n_correct = 0
    for i in tqdm(range(NUM_TEST)):
        empty = True
        while empty:
            img_path = random.choice(list(d_test.keys()))
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            name = img_path.split('/')[-1].split('.')[0]
            caps = d_test[img_path]
            if len(caps) > 0:
                pos_txt = random.choice(caps)
                #         pos_txt = ' '.join(pos_txt)
                empty = False
            print(pos_txt)
        neg_i = 0
        neg_txts = []
        while neg_i < NUM_NEG:
            img_path = random.choice(list(d_test.keys()))
            neg_name = img_path.split('/')[-1].split('.')[0]
            if neg_name == name:
                continue
            caps = d_test[img_path]
            if len(caps) == 0:
                continue
            neg_txt = random.choice(caps)
            if neg_txt in neg_txts:
                continue
            neg_txts.append(neg_txt)
            neg_i += 1
            print(name)
            print(f"Positive caption: {pos_txt}")
            print(f"Negative caption: {neg_txts}")
        text = clip.tokenize([pos_txt] + neg_txts).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            print("Label probs:", probs)
            print(np.argmax(probs))
        if np.argmax(probs) == 0:
            n_correct += 1
    print(f"Test precision {n_correct / NUM_TEST}")
