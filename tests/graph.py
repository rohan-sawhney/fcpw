#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

x_axis_types = ['Type', 'Vectorized', 'Coherent', 'Heuristic', 'Threads']
y_axis_types = ['Nodes', 'Primitives', 'Time']

x_axis_values = [['CPQ','RAY'],['no','yes'],['no','yes'],['Bvh_LongestAxisCenter','Bvh_OverlapSurfaceArea','Bvh_SurfaceArea','Bvh_OverlapVolume','Bvh_Volume'],['1','2','4','8','16','32','64','128']]

parser = argparse.ArgumentParser(description='Graph FCPQ benchmark data.')

parser.add_argument('data', help='data file')
parser.add_argument('rays', help='benchmark rays')
parser.add_argument('vectorize', help='use vectorized bvh')
parser.add_argument('coherent', help='use coherent queries')
parser.add_argument('heuristic', help='build heuristic')
parser.add_argument('threads', help='number of threads')
parser.add_argument('x', help='xaxis')
parser.add_argument('y', help='yaxis')
parser.add_argument('save', nargs='?', help='output file path', default='')
args = parser.parse_args()

if args.rays not in x_axis_values[0]:
    print('invalid rays arg: ', x_axis_values[0])
if args.vectorize not in x_axis_values[1]:
    print('invalid vectorize arg: ', x_axis_values[1])
if args.coherent not in x_axis_values[2]:
    print('invalid coherent arg: ', x_axis_values[2])
if args.heuristic not in x_axis_values[3]:
    print('invalid heuristic arg: ', x_axis_values[3])
if args.threads not in x_axis_values[4]:
    print('invalid threads arg: ', x_axis_values[4])
if args.x not in x_axis_types:
    print('invalid x arg: ', x_axis_values)
if args.y not in y_axis_types:
    print('invalid y arg: ', y_axis_types)

file = open(args.data, mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()

db = {}
for line in lines:
    line = line.split(',')
    line = [i.strip() for i in line]
    if line[0] in x_axis_values[0]:
        db[tuple(line[0:5])] = line[5:]

x_axis_idx = x_axis_types.index(args.x)
labels = x_axis_values[x_axis_idx]
values = []

key = [args.rays,args.vectorize,args.coherent,args.heuristic,args.threads]
for k in labels:
    key[x_axis_idx] = k
    try:
        values.append(db[tuple(key)][y_axis_types.index(args.y)])
    except:
        pass

labels = labels[0:len(values)]

plt.title('FCPQ Benchmark: {} vs {}'.format(args.x,args.y))
plt.plot(labels, values)
plt.xlabel(args.x)
plt.ylabel(args.y)
plt.gca().invert_yaxis()

if args.save == '':
    plt.show()
else:
    plt.savefig(args.save + '/{}_{}.png'.format(args.x,args.y))
