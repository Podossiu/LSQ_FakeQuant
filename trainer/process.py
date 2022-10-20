import logging
import math
import operator
import time
from quan import *
import torch

from util import AverageMeter, save_checkpoint_quantized

__all__ = ['train_qat', 'validate', 'PerformanceScoreboard']
#torch.backends.quantized.engine = 'qnnpack'
logger = logging.getLogger()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_qat(train_loader, val_loader, test_loader,qat_model, 
        criterion, optimizer, lr_scheduler, epoch, monitors, 
        args, init_qparams = True):

    start_epoch = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.n_gpu > 1:
        qat_model = torch.nn.DataParallel(qat_model)
    
    qat_model.to(args.device_type)

    criterion = criterion
    optimizer = optimizer
    lr_scheduler = lr_scheduler
    
    perf_scoreboard = PerformanceScoreboard(args.log_num_best_scores)
    
    quantized_model = None
    if args.resume_path or args.pre_trained:
        logger.info('>>>>>> Epoch -1 (pre-trained model evaluation)')
        top1, top5, _ = validate(val_loader, qat_model, criterion, start_epoch - 1, monitors, args)
        perf_scoreboard.update(top1, top5, start_epoch - 1)
    for epoch in range(start_epoch, args.epochs):
        logger.info('>>>>>> Epoch %3d' %epoch)
        t_top1, t_top5, t_loss = train_one_epoch_qat(train_loader, qat_model, criterion, 
                optimizer, lr_scheduler, epoch,monitors, args, init_qparams = init_qparams)

        quantized_model = torch.ao.quantization.convert(qat_model.cpu().eval(), inplace = False).to("cpu")
        print("validation quantized model on cpu")
        quantized_model.eval()
        qat_model.to(args.device_type)
        v_top1, v_top5, v_loss = validate(val_loader, qat_model.eval(), criterion, epoch, monitors, args, quantized = False)
        print(v_top1, v_top5)
        v_top1, v_top5, v_loss = validate(val_loader, quantized_model, criterion, epoch, monitors, args, quantized = True)
        #tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
        #tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
        #tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

        perf_scoreboard.update(v_top1, v_top5, epoch)
        is_best = perf_scoreboard.is_best(epoch)
        
        save_checkpoint_quantized(epoch, args.arch,qat_model, quantized_model, {'top1' : v_top1, 'top5' : v_top5}, is_best, args.name, args.log_dir)
    logger.info('>>>>>> Epoch -1 (final model evaluation)')
    validate(test_loader, quantized_model, criterion, -1, monitors, args, quantized = True)

    #tbmonitor.writer.close()
    logger.info('Program completed sucessfully ... exiting ...')

def train_one_epoch_qat(train_loader, qat_model, criterion, optimizer, lr_scheduler, epoch, monitors, args, init_qparams):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    
    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    qat_model.train()
    end_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if epoch == 0 and init_qparams == True and batch_idx == 0:
            qat_model.apply(init_mode)

        inputs = inputs.to(args.device_type)
        targets = targets.to(args.device_type)
        outputs = qat_model(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if epoch == 0 and init_qparams == True and batch_idx == 0:
            qat_model.apply(training_mode)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if (batch_idx + 1) % args.log_print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr'],
                })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, epoch, monitors, args, quantized = False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            if quantized:
                inputs = inputs.to("cpu")
                targets = targets.to("cpu")
            else:
                inputs = inputs.to(args.device_type)
                targets = targets.to(args.device_type)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.log_print_freq == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch