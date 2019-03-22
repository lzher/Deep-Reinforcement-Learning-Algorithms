import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import sys

parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
parser.add_argument('-b', '--bn', default='05', type=str, help='BASE result number')
parser.add_argument('-p', '--pn', default='05', type=str, help='PER result number')
parser.add_argument('-n', '--name', default='rewards', type=str, help='PER result number')
args = parser.parse_args()


b = sio.loadmat('logs/BASE_%s/%s.mat' % (args.bn, args.name))[args.name]
p = sio.loadmat('logs/PER_%s/%s.mat' % (args.pn, args.name))[args.name]

plt.plot(b.mean(0))
plt.plot(p.mean(0))
plt.legend(['BASE', 'PER'])
plt.grid(True)
plt.show()

