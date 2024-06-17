import numpy as np
import os
from Optim import *
from Model import *
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--epochs', type=int, default=300, help='epochs to run [default: 10]')
parser.add_argument('--seed', type=int, default=42, help='Random seed [default: 42]')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate [default: 1e-5]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size [default: 128]')
parser.add_argument('--num_classes', type=int, default=5, help='The number of classes [default: 5]')
parser.add_argument('--model_name', type=str, default='ResNet34', help='Call model name [default: ResNet34]')

args = parser.parse_args()
device = torch.device('cuda:' + '{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available() is not True:
    print("GPU is not available.")
torch.manual_seed(args.seed)

Trainer = Optim(device, args.lr, args.num_classes, args.epochs, args.batch_size, args.model_name)
load = os.path.join('Parameters/'+'resnet', args.model_name+".pth")

if args.model_name == 'ResNet18':
    model = ResNet18(args.num_classes)
elif args.model_name == 'ResNet34':
    model = ResNet34(args.num_classes)
elif args.model_name == 'ResNet50':
    model = ResNet50(args.num_classes)
elif args.model_name == 'ModifiedResNet':
    model = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=args.num_classes, heads=16, input_resolution=224, width=64)
else:
    raise ValueError(f"Unknown model name: {args.model_name}")

model = model.to(device)
train_path = './data/flower_photos'
try:
    model.load_state_dict(torch.load(load))
    model, history = Trainer.Train(train_path, model)
except:
    model, history = Trainer.Train(train_path)

