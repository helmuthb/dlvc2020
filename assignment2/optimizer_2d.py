import os
import time
import math
import heapq
from collections import namedtuple

import cv2
import torch
import numpy as np

Vec2 = namedtuple('Vec2', ['x1', 'x2'])

class AutogradFn(torch.autograd.Function):
    '''
    This class wraps a Fn instance to make it compatible with PyTorch optimimzers
    '''
    @staticmethod
    def forward(ctx, fn, loc):
        ctx.fn = fn
        ctx.save_for_backward(loc)
        value = fn(Vec2(loc[0].item(), loc[1].item()))
        return torch.tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        loc, = ctx.saved_tensors
        grad = fn.grad(Vec2(loc[0].item(), loc[1].item()))
        return None, torch.tensor([grad.x1, grad.x2]) * grad_output

class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str, eps: float):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self.fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self.fn = self.fn.astype(np.float32)
        self.fn /= (2**16-1)
        self.eps = eps

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''

        # convert from [0,1] to [0,255]
        img = self.fn * 256
        img = img.astype(np.uint8)
        # using cv2.applyColorMap as the hint suggested
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        fn = self.fn
        max_x1 = fn.shape[0]-1
        max_x2 = fn.shape[1]-1
        if loc.x1 < 0 or loc.x1 > max_x1 or loc.x2 < 0 or loc.x2 > max_x1:
            raise ValueError("Location is out of bounds")
        # using bilinear interpolation
        # formula taken from Wikipedia
        i1 = math.floor(loc.x1)
        i2 = math.floor(loc.x2)
        x1 = loc.x1 - i1
        x2 = loc.x2 - i2
        f00 = fn[i1,i2]
        f10 = fn[i1+1,i2] if i1 < max_x1 else f00
        f01 = fn[i1,i2+1] if i2 < max_x2 else f00
        if i1 == max_x1:
            f11 = f01
        elif i2 == max_x2:
            f11 = f10
        else:
            f11 = fn[i1+1,i2+1]
        return f00*(1-x1)*(1-x2) + f10*x1*(1-x2) + f01*(1-x1)*x2 + f11*x1*x2

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        eps = self.eps
        if eps <= 0: raise ValueError("Epsilon is too small")
        max_x1 = float(self.fn.shape[0]-1)
        max_x2 = float(self.fn.shape[1]-1)
        if loc.x1 < 0 or loc.x1 >= max_x1 or loc.x2 < 0 or loc.x2 >= max_x2:
            raise ValueError("Location is out of bounds")
        # if we are away from border, I use X-eps/X+eps, delta=2*eps
        # if we are at the border I use the point loc as one of the locations
        delta_x1 = 2*eps
        if loc.x1 > eps:
            loc1 = Vec2(loc.x1-eps, loc.x2)
        else:
            loc1 = Vec2(loc.x1, loc.x2)
            delta_x1 = eps
        if loc.x1 + eps <= max_x1:
            loc2 = Vec2(loc.x1+eps, loc.x2)
        else:
            loc2 = Vec2(loc.x1, loc.x2)
            delta_x1 = eps
        # numerical gradient - component 1
        grad_x1 = (self(loc2) - self(loc1)) / delta_x1
        # Same thing for the other dimension
        delta_x2 = 2*eps
        if loc.x2 > eps:
            loc1 = Vec2(loc.x1, loc.x2-eps)
        else:
            loc1 = Vec2(loc.x1, loc.x2)
            delta_x2 = eps
        if loc.x2 + eps <= max_x2:
            loc2 = Vec2(loc.x1, loc.x2+eps)
        else:
            loc2 = Vec2(loc.x1, loc.x2)
            delta_x2 = eps
        # numerical gradient - component 1
        grad_x2 = (self(loc2) - self(loc1)) / delta_x2
        # return gradient
        return Vec2(grad_x1, grad_x2)

if __name__ == '__main__':
    # Parse args
    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=10.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # Init
    fn = Fn(args.fpath, args.eps)
    vis = fn.visualize()
    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)
    # priority queue: top n results
    best_n = []
    # breakpoint n: when the value is no longer in the top-n
    n = 20
    # optimizer = torch.optim.SGD([loc], lr=args.learning_rate, momentum=args.beta, nesterov=args.nesterov)
    optimizer = torch.optim.AdamW([loc], lr=args.learning_rate, eps=args.eps, weight_decay=0)

    # Perform gradient descent using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use it
    while True:
        old = (int(loc[0].item()), int(loc[1].item()))
        optimizer.zero_grad()
        value = AutogradFn.apply(fn, loc)
        # push value on heap
        if len(best_n) >= n:
            worst = heapq.heappushpop(best_n, -value)
            if (-value) == worst:
                # break the loop
                exit(0)
        else:
            heapq.heappush(best_n, -value)
        value.backward()
        optimizer.step()
        print(loc[0].item(), loc[1].item())
        new = (int(loc[0].item()), int(loc[1].item()))
        # Visualize each iteration by drawing on vis
        cv2.line(vis, old, new, (255, 255, 255), 2)
        cv2.imshow('Progress', vis)
        cv2.waitKey(5)  # 20 fps, tune according to your liking