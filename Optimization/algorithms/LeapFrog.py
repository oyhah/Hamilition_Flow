import torch

import functions

def Leap_Frog(x, optimizer, args):

    num_iter = args.K
    theta = args.theta

    if args.function == 'Ackley':
        func = functions.Ackley

    v = torch.zeros_like(x)

    for iter in range(num_iter):

        optimizer.zero_grad()
        loss = func(x)
        loss.backward()
        
        v = v - theta / 2 * x.grad

        if args.norm == 'l2':
            x.data = x.data + theta * v
        elif args.norm == 'l1':
            x.data = x.data + theta * torch.sign(v)
        elif args.norm == 'normalized':
            x.data = x.data + theta * v / torch.linalg.norm(v)
        elif args.norm == 'coordinate':
            index = torch.argmax(torch.abs(v))
            v_index = torch.zeros_like(v)
            v_index[index] = torch.sign(v[index])
            x.data = x.data + theta * v_index

        optimizer.zero_grad()
        loss = func(x)
        loss.backward()

        v = v - theta / 2 * x.grad

    return x