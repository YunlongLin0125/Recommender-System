import lib
import numpy as np
import torch
from tqdm import tqdm

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        dataloader = lib.DataLoader(eval_data, batch_size)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            #for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)
                logit, hidden = self.model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                recall, mrr = lib.evaluate(logit, target, k=self.topk)

                # torch.Tensor.item() to get a Python number from a tensor containing a single value
                # losses.append(loss.item())
                recalls.append(recall)
                # mrrs.append(mrr)
                losses.append(loss.cpu().item())
                recalls.append(recall)
                mrrs.append(mrr.cpu().item())

        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)

        return mean_losses, mean_recall, mean_mrr

    def hit_ratio_at_k(self, rankings, k):
        return 1.0 if any(rank < k for rank in rankings) else 0.0

    def ndcg_at_k(self, rankings, k):
        return sum(1.0 / np.log2(rank + 2) for rank in rankings if rank < k)

    def eval_SASRec(self, eval_data, batch_size, k):
        self.model.eval()
        losses = []
        hit_ratios = []
        ndcgs = []
        dataloader = lib.DataLoader(eval_data, batch_size)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask) in tqdm(enumerate(dataloader),
                                                  total=len(dataloader.dataset.df) // dataloader.batch_size,
                                                  miniters=1000):
                input = input.to(self.device)
                target = target.to(self.device)
                logit, hidden = self.model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)

                # Compute rankings
                rankings = (-logit).argsort(dim=1).cpu().numpy()
                target_rankings = np.array([np.argwhere(row == target[i].item()) for i, row in enumerate(rankings)])

                # Compute Hit Ratio@k and NDCG@k
                hit_ratio = self.hit_ratio_at_k(target_rankings, k)
                ndcg = self.ndcg_at_k(target_rankings, k)

                losses.append(loss.cpu().item())
                hit_ratios.append(hit_ratio)
                ndcgs.append(ndcg)

        mean_losses = np.mean(losses)
        mean_hit_ratios = np.mean(hit_ratios)
        mean_ndcgs = np.mean(ndcgs)

        return mean_losses, mean_hit_ratios, mean_ndcgs