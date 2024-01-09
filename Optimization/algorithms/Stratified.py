import torch
import copy

import functions

def Stratified_Integrator(x, optimizer, args):

    num_iter = args.K
    theta = args.theta

    if args.function == 'Ackley':
        func = functions.Ackley

    v = torch.zeros_like(x)

    for iter in range(num_iter):

        u = torch.rand(1) * theta

        x_s = copy.deepcopy(x.data)


        if args.norm == 'l2':
            x.data = x.data + u * v
        elif args.norm == 'l1':
            x.data = x.data + u* torch.sign(v)
        elif args.norm == 'normalized':
            x.data = x.data + u * v / torch.linalg.norm(v)
        elif args.norm == 'coordinate':
            index = torch.argmax(torch.abs(v))
            v_index = torch.zeros_like(v)
            v_index[index] = torch.sign(v[index])
            x.data = x.data + u * v_index
        
        optimizer.zero_grad()
        loss = func(x)
        loss.backward()
        
        v_half = v - theta / 2 * x.grad
        v = v - theta * x.grad

        x.data = x_s

        if args.norm == 'l2':
            x.data = x.data + theta * v_half
        elif args.norm == 'l1':
            x.data = x.data + theta * torch.sign(v_half)
        elif args.norm == 'normalized':
            x.data = x.data + theta * v_half / torch.linalg.norm(v_half)
        elif args.norm == 'coordinate':
            index = torch.argmax(torch.abs(v_half))
            v_half_index = torch.zeros_like(v_half)
            v_half_index[index] = torch.sign(v_half[index])
            x.data = x.data + theta * v_half_index

    return x