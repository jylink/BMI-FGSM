import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bmifgsm import BMIFGSM
from model import load_net
from utils import get_preprocess_deprocess, get_bounds


if __name__ == '__main__':
    device = torch.device('cuda')
    net = load_net(model='vgg16', pretrained=True).to(device)
    net.eval()

    preprocess, deprocess = get_preprocess_deprocess("imagenet", size="imagenet")

    img_path = 'sample.jpg'
    img_raw = Image.open(img_path)
    img_tensor = preprocess(img_raw)
    bounds = get_bounds(preprocess(img_raw), "imagenet")

    #################################
    EPS = [0.01 for i in range(10)]
    T = [i * 100 for i in range(10)]
    MU = 0
    N = 100
    MAX_ITR = 1000
    F = 1
    CR = 0.9
    KR = 0.1
    EVO_STRATEGY = 'sign'
    EVA_STRATEGY = 'cw'
    #################################

    attacker = BMIFGSM(net=net, bounds=bounds, EPS=EPS, T=T, mu=MU, n=N, max_itr=MAX_ITR, F=F, CR=CR, KR=KR,
                    evolve_strategy=EVO_STRATEGY, evaluate_strategy=EVA_STRATEGY, reset=True, debug=True, 
                    early_stop=True, check_stop=10, device=device)

    img = img_tensor.data.numpy()
    ground_truth = attacker.predict(img_tensor.to(device), 0)[1]

    adv = attacker.attack(img, ground_truth)
    adv_tensor = torch.from_numpy(adv).to(device)

    deprocess(img_tensor).save('original.jpg')
    deprocess(adv_tensor).save('adversarial.jpg')



    