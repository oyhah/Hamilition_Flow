import random
import torch
import numpy as np

from setting import args_parser
from logg import get_logger
import functions
from algorithms.GD import Gradient_Descent
from algorithms.LeapFrog import Leap_Frog
from algorithms.Stratified import Stratified_Integrator
from data_set import set_data
from models import model_net
from resnet import ResNet18
from test import test

import torch.multiprocessing as mp
import torch.nn as nn

def ParallelInnerLoop(args, x_net, train_dataloader):

    mp.set_start_method('spawn', force=True)
    queues = mp.Queue(), mp.Queue(), mp.Queue()

    processes = []
    num_processes = torch.cuda.device_count()

    for rank in range(-1, num_processes):
        p = mp.Process(target=run, args=(args, rank, num_processes, queues, x_net, train_dataloader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()    # wait for all subprocesses to finish
    
    x_net_update = queues[2].get()
    x_net.load_state_dict(x_net_update.state_dict())
    
    return x_net


def run(args, rank, num_processes, queues, x_net, train_dataloader):

    loss_func = nn.CrossEntropyLoss()
    lr = (args.theta)**2
    threshold = 0.1

    if rank == -1:
        device = f'cuda:0'
        # Model
        nets = [ResNet18().to(device) for i in range(num_processes + 1)]
        for i in range(num_processes):
            nets[i].load_state_dict(x_net.state_dict())

        # Parallel training        
        window_start = 0
        ite = 0
        while window_start < args.K:
            for i in range(num_processes):
                queues[0].put( (nets[i], i) )
            
            grad_all = [None for _ in range(num_processes)]

            for i in range(num_processes):
                info = queues[1].get()
                grad_rec, id = info
                grad_all[id] = grad_rec.to(device)
                del info
            
            torch.cuda.synchronize()
            
            error_window = torch.zeros(num_processes)
            for i in range(1, num_processes + 1):
                for name, param in nets[i].named_parameters():
                    grad = 0
                    for j in range(i):
                        grad += (i - j) * grad_all[j][name]
                    param.data = nets[0][name] - lr * grad
                    error_window[i - 1] = torch.max( lr * torch.norm(torch.abs(grad)) / torch.numel(grad), error_window[i] )
            
            (id_error,) = torch.where(error_window > threshold)

            if id_error.numel() == 0:
                stride = num_processes
            else:
                stride = torch.min(id_error)

            for  i in range(num_processes - stride + 1):
                nets[i].load_state_dict(nets[i + stride].state_dict())
            
            net_state = nets[num_processes].state_dict()
            for i in range(num_processes - stride + 1, num_processes + 1):
                nets[i].load_state_dict(net_state)

            window_start += stride
            ite += 1
        
        # Finish the inner loop training
        for _ in range(num_processes):
            queues[0].put(None)
        
        queues[2].put(nets[num_processes - 1])
            
    else:
        device = f'cuda:{rank}'
        net_local = ResNet18().to(device)
        optimizer_local = torch.optim.SGD(net_local.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weigh_delay)

        train_iter = train_dataloader.__iter__()

        while True:
            info = queues[0].get()
            if info is None:
                del info
                return
            
            net_rec, id = info

            net_local.load_state_dict(net_rec.state_dict())

            inputs, targets = train_iter.__next__()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer_local.zero_grad()
            outputs = net_local(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()

            grad_local = {name: param.grad for name, param in net_local.named_parameters()}

            del info
            queues[1].put( (grad_local, id) )




if __name__ == '__main__':

    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Dataset
    train_dataloader, test_dataloader = set_data(args)

    # Model
    x_net = ResNet18()
    x_net = x_net.to(args.device)


    # Training
    logger = get_logger(args.filepath)
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')

    logger.info('start training!')

    loss_results = []
    acc_results = []

    for iter in range(args.epochs + 1):

        # Training

        x_net  = ParallelInnerLoop(args, x_net, train_dataloader)

        # Test
        
        test_loss, test_acc = test(x_net, test_dataloader, args)

        logger.info('Epoch:[{}]\tlr =\t{:.5f}\ttest_loss=\t{:.5f}\ttest_acc=\t{:.5f}'.
                            format(iter, args.lr, test_loss, test_acc))

        loss_results.append(test_loss)
        acc_results.append(test_acc)
    

    logger.info('finish training!')

    loss_results = np.array(loss_results)
    acc_results = np.array(acc_results)

    np.save('result/loss_%s_%s_%s_theta%.5f_K%d_momentum%.3f.npy' % (args.dataset, args.method, args.norm, args.theta, args.K, args.momentum), loss_results)
    np.save('result/acc_%s_%s_%s_theta%.5f_K%d_momentum%.3f.npy' % (args.dataset, args.method, args.norm, args.theta, args.K, args.momentum), acc_results)

