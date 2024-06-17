import os
from Optim import *
from Model import *
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
parser.add_argument('--epochs', type=int, default=100, help='epochs to run [default: 10]')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_classes', type=int, default=5, help='')
parser.add_argument('--model_name', type=str, default='resnet18', help='')

args = parser.parse_args()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:' + '{}'.format(args.gpu))
# device = torch.device('cpu')
torch.manual_seed(args.seed)
model = ResNet18(args.num_classes)

train_path = '../data/flower'

load = os.path.join('Parameters/'+'resnet', args.model_name+".pth")
model.load_state_dict(torch.load(load))

