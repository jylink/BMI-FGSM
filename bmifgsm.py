import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aliyun import alinet, alinet_single


class BMIFGSM:
    
    def __init__(self, net, bounds, EPS, T, mu=1, n=100, max_itr=1000, F=0.501, CR=1, KR=0, 
                 evolve_strategy='clip', evaluate_strategy='conf', reset = False, eps_evo_func=None,
                 debug=True, early_stop=True, check_stop=10, targeted=False, device=None):
        """
        T[0] must be 0
        EPS[0] the init epsilon (epsilon for perturbation)
        The final |r|_infty will be sum(EPS)

        T: list of int
        EPS: list of float
        """
        assert n >= 3
        assert len(T) == len(EPS)
        assert T[0] == 0
        
        self.net = net
        self.clip_min, self.clip_max = map(np.array, bounds)
        self.EPS = list(EPS)
        self.T = list(T)
        self.mu = mu
        self.n_candidates = n
        self.max_itr = max_itr
        self.F = F
        self.CR = CR
        self.KR = KR
        
        self.evolve_strategy = evolve_strategy
        self.evaluate_strategy = evaluate_strategy
        self.reset = reset
        self.eps_evo_func = eps_evo_func
        
        self.debug = debug
        self.early_stop = early_stop
        self.check_stop = check_stop
    
        self.targeted = targeted
        self.device = device


    def predict(self, img, target):
        """
        img: tensor
        """
        if self.net == 'aliyun':
            img = deprocess(img.cpu().clone())
            result = alinet_single(img)
            result = json.loads(result)
            tags = ""
            s = 0
            for i in result["tags"]:
                if i["value"] in target:
                    s += i["confidence"] / 100
                tags += "({},{})".format(i["value"], i["confidence"] / 100)
            score = s
            return score, tags, -1
        else:
            out = self.net(img.unsqueeze(0))
            softmaxs = F.softmax(out, dim=1)
            conf, pred = softmaxs.max(dim=1)
            target_conf = softmaxs[0, target]
            return conf.item(), pred.item(), target_conf.item()


    def init(self, *size):
        """
        size: d0, d1, d2 ...
        return: ndarray, (n, *size)
        """
        candidates = np.random.randn(self.n_candidates, *size)

        if self.evolve_strategy == 'sign':
            candidates = np.sign(candidates)

        return candidates


    def evolve(self, candidates, fitness):
        """
        candidates: ndarray, (n, *size)
        fitness: ndarray, (n,)
        return: ndarray, (n, *size)
        """
        n = candidates.shape[0]

        indice = np.zeros((n, 3)).astype(np.int)
        
        for i in range(n):
            while True:
                a = np.random.randint(0, n, size=3)
                if not (a[0] == a[1] or a[1] == a[2] or a[2] == a[0] or (i in a)):
                    break
            indice[i] = a

        # evolve
        next_gen = candidates[indice[:, 0]] + self.F * (candidates[indice[:, 1]] - candidates[indice[:, 2]])

        # apply evolve strategy to next generation
        if self.evolve_strategy == 'clip':
            D = 1
            next_gen = np.clip(next_gen, -D, D)
            
        elif self.evolve_strategy == 'sign':
            next_gen = np.sign(next_gen)
            
        elif self.evolve_strategy == 'none':
            pass
        else:
            assert 0

        # crossover
        if self.CR < 1:
            cr_mask = np.random.rand(*next_gen.shape)
            cr_indice = np.where(cr_mask > self.CR)
            next_gen[cr_indice] = candidates[cr_indice]
            
        return next_gen


    def perturb(self, img, candidates, epsilon, g):
        """
        img: ndarray, (c, h, w)
        candidates: ndarray, (n, c, h, w)
        return: ndarray, (n, c, h, w)
        """
        n = candidates.shape[0]

        # apply perturbation to image
        _g = self.mu * g + candidates / np.sum(np.abs(candidates).reshape(n, -1), axis=1)[:, None, None, None]
        
        pert_imgs = np.clip(img + epsilon * np.sign(_g), self.clip_min, self.clip_max)

        return pert_imgs.astype(np.float32)


    def evaluate(self, imgs, target):
        """
        LOCAL
        imgs: ndarray, (n, c, h, w)
        return: ndarray, (n,)

        ALIYUN
        imgs: ndarray, (n, c, h, w)
        target: list, [cleanTag1, cleanTag2, ...]
        return: ndarray, (n,)
        """
        if self.net == 'aliyun':
            return alinet(imgs, target)

        imgs = torch.from_numpy(imgs).to(self.device)

        if self.evaluate_strategy == 'conf':
            out = self.net(imgs)
            out = F.softmax(out, dim=1)  # (n, #class)
            if self.targeted:
                confs = -out[:, target]
            else:
                confs = out[:, target]
            return confs.cpu().data.numpy().copy().reshape(-1)

        elif self.evaluate_strategy == 'cw':
            out = self.net(imgs)
            reals = out[:, target].clone()
            out[:, target] = -1e4
            others = out.max(dim=1)[0]
            if self.targeted:
                loss = others - reals
            else:
                loss = reals - others
            return loss.cpu().data.numpy().copy().reshape(-1)

        else:
            assert 0


    def attack(self, img, target):
        """
        img: ndarray, (c, h, w)
        target: scalar, true label OR target label if targeted
        """

        def show_info():
            best_adv = torch.from_numpy(get_best_adv()[0]).to(self.device)
            conf, pred, target_conf = self.predict(best_adv, target)
            if self.debug:
                print('itr: {:4} | #successor: {:4} | best fitness: {:.4f} | prediction: {:.4f}, {} | target conf: {:.4f}'.format(
                itr, n_successor, best_target_conf, conf, pred, target_conf))
            return [pred]

        def get_best_adv():
            pert_imgs = self.perturb(img, candidates, eps_pert, g)
            idx = np.argmin(fitness)
            best_adv = pert_imgs[idx]
            return best_adv.copy(), idx

        assert isinstance(img, np.ndarray)
        img = img.copy()
        g = np.zeros_like(img)
        
        # init candidates & fitness
        candidates = self.init(*img.shape)
        fitness = self.evaluate(np.expand_dims(img, 0), target)
        fitness = np.full(self.n_candidates, fitness[0])

        # init epsilon
        self.eps_evo_func = self.eps_evo_func or (lambda x: x)
        eps_itr = 0
        eps_pert = self.EPS[eps_itr]  # for perturbation
        eps_evo = sum(self.EPS)  # for evolve
        eps_itr += 1

        list_best_target_conf = []
        list_n_successor = []
        list_eps_pert = []
        list_eps_evo = []
        list_grad = []
        list_g = []

        for itr in range(self.max_itr):
            
            # evolve, evaluate
            next_gen = self.evolve(candidates, fitness)
            pert_imgs = self.perturb(img, next_gen, self.eps_evo_func(eps_evo), g)
            next_fit = self.evaluate(pert_imgs, target)

            # select, replace
            successor = next_fit < fitness
            candidates[successor] = next_gen[successor]
            fitness[successor] = next_fit[successor]

            best_target_conf = fitness.min()
            n_successor = np.sum(successor)

            list_best_target_conf.append(best_target_conf)
            list_n_successor.append(n_successor)
            list_eps_pert.append(eps_pert)
            list_eps_evo.append(self.eps_evo_func(eps_evo))

            # early stop
            if itr % (self.max_itr // self.check_stop) == 0 or itr == self.max_itr - 1:
                preds = show_info()
                if self.early_stop and ((not self.targeted and not target in preds) or (self.targeted and target in preds)):
                    if self.debug: print('early stop')
                    break
                    
            # update epsilon and adv
            if eps_itr < len(self.T) and itr == self.T[eps_itr]:
                img, idx = get_best_adv()

                if itr < self.max_itr - 1:
                    g = self.mu * g + candidates[idx] / np.sum(np.abs(candidates[idx]))
                    list_g.append(g.copy())
                    list_grad.append(candidates[idx].copy())

                    # keep the first KR*100% best candidates and reset others
                    if self.reset:
                        indice = np.argsort(fitness)[int(len(fitness) * self.KR):]
    
                        candidates_ = self.init(*img.shape)
                        fitness_ = self.evaluate(np.expand_dims(img, 0), target)
                        fitness_ = np.full(self.n_candidates, fitness_[0])

                        candidates[indice] = candidates_[indice]
                        fitness[indice] = fitness_[indice]

                eps_evo -= eps_pert
                eps_pert = self.EPS[eps_itr]
                eps_itr += 1

        img, _ = get_best_adv()
        return img