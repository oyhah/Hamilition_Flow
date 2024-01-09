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
from test import test

if __name__ == '__main__':

    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Dataset
    train_dataloader, test_dataloader = set_data(args)

    # Model
    x_net = model_net(args.dataset)
    x_net = x_net.to(args.divice)


    # Training
    logger = get_logger(args.filepath)
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')

    logger.info('start training!')

    optimizer = torch.optim.SGD(x_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weigh_delay)

    for iter in range(args.epochs + 1):

        # Training

        if args.method == 'GD':
            x = Gradient_Descent(x_net, optimizer, train_dataloader, args)
        elif args.method == 'LF':
            x = Leap_Frog(x_net, optimizer, train_dataloader, args)
        elif args.method == 'SI':
            x = Stratified_Integrator(x_net, optimizer, train_dataloader, args)

        # Test
        
        test_loss, test_acc = test(x_net, test_dataloader, args)

        logger.info('Epoch:[{}]\tlr =\t{:.5f}\ttest_loss=\t{:.5f}\ttest_acc=\t{:.5f}'.
                            format(iter, args.lr, test_loss, test_acc))
    

    logger.info('finish training!')

