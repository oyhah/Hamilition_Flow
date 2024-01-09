import torch

import functions

def Gradient_Descent(x, optimizer):

    optimizer.zero_grad()
    loss = functions.Ackley(x)
    loss.backward()
    optimizer.step()

    return x