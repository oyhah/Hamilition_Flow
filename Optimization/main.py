import random
import torch
import numpy as np

from setting import args_parser
from logg import get_logger
import functions
from algorithms.GD import Gradient_Descent
from algorithms.LeapFrog import Leap_Frog
from algorithms.Stratified import Stratified_Integrator

if __name__ == '__main__':

    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Function
    if args.function == 'Ackley':
        func = functions.Ackley


    # Model
    x_tensor = torch.FloatTensor([1, 1])
    x = torch.autograd.Variable(x_tensor, requires_grad=True)


    # Training
    logger = get_logger(args.filepath)
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')

    logger.info('start training!')

    optimizer = torch.optim.SGD([x], lr=args.lr, momentum=args.momentum, weight_decay=args.weigh_delay)

    for iter in range(args.epochs + 1):

        # Training
        if args.method == 'GD':
            x = Gradient_Descent(x, optimizer, args)
        elif args.method == 'LF':
            x = Leap_Frog(x, optimizer, args)
        elif args.method == 'SI':
            x = Stratified_Integrator(x, optimizer, args)

        # Test
        f_value = func(x)

        logger.info('Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\t'.
                            format(iter, args.lr, f_value))
    

    logger.info('finish training!')

    print(x)

