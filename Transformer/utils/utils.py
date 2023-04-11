'''
Author: Felix
Date: 2023-04-11 15:13:46
LastEditors: Felix
LastEditTime: 2023-04-11 16:48:42
Description: tools for training and testing
'''
from tqdm import tqdm
import torch
import os


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                        for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                            for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class Trainer:

    def __init__(self, model, train_loader, validate_loader, export_path, device, optimizer, loss_fn, lr_scheduler, num_epochs) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = validate_loader
        self.export_path = export_path
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.metric_ks = [1, 5, 10, 20, 50, 100]

    def train_one_step(self, epoch):
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]
            x, y = batch
            #  calcuate logits
            logits = self.model(x)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            loss = self.loss_fn(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            # if batch_idx % 400 == 0:
            #     loss, current = loss.item(), batch_idx
            #     print(f"loss: {loss:>7f}  [{current:>5d}]")
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.6f} '.format(epoch+1, total_loss/(batch_idx+1)))
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def validate(self, epoch):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + \
                    ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace(
                    'NDCG', 'N').replace('Recall', 'R')
                description = description.format(
                    *(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

    def train(self):
        # self.validate(0)
        for epoch in range(self.num_epochs):
            self.train_one_step(epoch)
            # self.validate(epoch)
            torch.save(self.model, os.path.join(
                self.export_path, str(epoch)+'.model'))

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass


# def train_one_step(model, optimizer, loss_fn, epoch, dataloader, device, lr_scheduler=None):
#     model.train()
#     if lr_scheduler:
#         lr_scheduler.step()
#     tqdm_dataloader = tqdm(dataloader)
#     average_meter_set = AverageMeterSet()
#     for batch_idx, batch in enumerate(tqdm_dataloader):
#         batch_size = batch[0].size(0)
#         batch = [x.to(device) for x in batch]
#         x, y = batch
#         optimizer.zero_grad()
#         #  calcuate logits
#         logits = model(x)
#         logits.view(-1, logits.size(-1))
#         y.view(-1)
#         loss = loss_fn(logits, y)
#         loss.backward()
#         optimizer.step()

#         average_meter_set.update('loss', loss.item())
#         tqdm_dataloader.set_description(
#             'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))


# def validate(model, epoch, accum_iter):
#     self.model.eval()

#     average_meter_set = AverageMeterSet()

#     with torch.no_grad():
#         tqdm_dataloader = tqdm(self.val_loader)
#         for batch_idx, batch in enumerate(tqdm_dataloader):
#             batch = [x.to(self.device) for x in batch]

#             metrics = self.calculate_metrics(batch)

#             for k, v in metrics.items():
#                 average_meter_set.update(k, v)
#             description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
#                                   ['Recall@%d' % k for k in self.metric_ks[:3]]
#             description = 'Val: ' + \
#                 ', '.join(s + ' {:.3f}' for s in description_metrics)
#             description = description.replace(
#                 'NDCG', 'N').replace('Recall', 'R')
#             description = description.format(
#                 *(average_meter_set[k].avg for k in description_metrics))
#             tqdm_dataloader.set_description(description)

#         log_data = {
#             'state_dict': (self._create_state_dict()),
#             'epoch': epoch+1,
#             'accum_iter': accum_iter,
#         }
#         log_data.update(average_meter_set.averages())
#         self.log_extra_val_info(log_data)
#         self.logger_service.log_val(log_data)
