from dataio import *
import sys
import os
from torch.utils.data import DataLoader
import argparse
import torch
from train import *
from model import *


p = argparse.ArgumentParser()

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# General training options
p.add_argument('--data_type',type=str,default='s')
p.add_argument('--batch_size', type=int, default=32000)
p.add_argument('--lr', type=float, default=1e-5, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=400,
               help='Number of epochs to train for.')
p.add_argument('--checkpoint', type=int, default=10,
               help='checkpoint is saved.')
p.add_argument('--dataset', type=str, default='Vortex',
               help='Scalar dataset; one of (Vortex, combustion)')
p.add_argument('--res', type=int, default=256,
               help='the resolution of rendering image')
p.add_argument('--model_path', type=str, default='../Exp/', metavar='N',
                    help='the path where we stored the saved model')
p.add_argument('--result_path', type=str, default='../Result/', metavar='N',
                    help='the path where we stored the synthesized data')
p.add_argument('--train', type=str, default='train', metavar='N',
                    help='the path where we stored the synthesized data')
p.add_argument('--application', type=str, default='temporal', metavar='N',
                    help='the path where we stored the synthesized data')
p.add_argument('--scale', type=int, default=4, metavar='N',
                    help='spatial upscaling factor')
p.add_argument('--interval', type=int, default=3, metavar='N',
                    help='temporal upscaling factor')
p.add_argument('--factor', type=int, default=4, metavar='N',
                    help='randomly sample factor*batch_size')
p.add_argument('--active', type=str, default='sine', metavar='N',
                    help='active function')
p.add_argument('--init', type=int, default=64, metavar='N',
                    help='init features')
p.add_argument('--hint', type=str, default='nosuper', metavar='N',
                    help='')
p.add_argument('--num_res', type=int, default=10, metavar='N',
                    help='number of residual blocks')
p.add_argument('--angle', type=int, default=15, metavar='N',
                    help='sampled angle')

opt = p.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()

def main():
  if opt.train == 'train':
    if opt.application in ['spatial','temporal','super-spatial']:
      Data = ScalarDataSet(opt)
      if opt.active == 'sine':
        print('Initalize Model Successfully using Sine Function!')
        Model = CoordNet(4,1,opt.init,opt.num_res)
      elif opt.active == 'relu':
        print('Initalize Model Successfully using ReLU Function!')
        Model = CoordNetReLU(4,1,opt.init,opt.num_res)
    elif opt.application == 'viewsynthesis':
      Data = ViewSynthesis(opt)
      Model = CoordNet(4,3,opt.init,opt.num_res)
    elif opt.application == 'AO':
      Data = AODataSet(opt)
      Model = CoordNet(4,1,opt.init,opt.num_res)
    Data.ReadData()
    Model.cuda()
    trainNet(Model,opt,Data)

  elif opt.train == 'inf':
    if opt.application in ['spatial','temporal','super-spatial','extrapolation']:
      Data = ScalarDataSet(opt)
    elif opt.application == 'viewsynthesis':
      Data = ViewSynthesis(opt)
    elif opt.application == 'AO':
       Data = AODataSet(opt)
    inf(Data,opt)

if __name__== "__main__":
    main()





