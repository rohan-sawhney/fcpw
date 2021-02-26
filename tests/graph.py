#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import argparse
from collections import defaultdict

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
if args.heuristic not in x_axis_values[3] and args.heuristic != 'avg':
    print('invalid heuristic arg: ', x_axis_values[3])
if args.threads not in x_axis_values[4] and args.threads != 'avg':
    print('invalid threads arg: ', x_axis_values[4])
if args.x not in x_axis_types and args.x != 'auto':
    print('invalid x arg: ', x_axis_values)
if args.y not in y_axis_types and args.y != 'auto':
    print('invalid y arg: ', y_axis_types)

file = open(args.data, mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()
db = {}
for line in lines:
    line = line.split(',')
    line = [i.strip() for i in line]
    if line[0] in x_axis_values[0]:
        data = [float(line[5]), float(line[6].strip('%')), float(line[7])]
        db[tuple(line[0:5])] = data


def accumulate(k, y_idx, ranges):
    acc = 0
    if len(ranges) > 0:
        i = 0
        for v in x_axis_values[ranges[0]]:
            k[ranges[0]] = v
            res = accumulate(k, y_idx, ranges[1:])
            if res != None: 
                acc += res
                i += 1
        if i > 0:
            return acc / i
        else:
            return None
    else:
        try:
            return db[tuple(k)][y_idx]
        except:
            return None

def do_graph(x_idx, y_idx):
    x_name = x_axis_types[x_idx]
    y_name = y_axis_types[y_idx]
    labels = x_axis_values[x_idx]
    values = []
    
    ranges = []
    if args.heuristic == 'avg' and x_name != 'Heuristic' and y_name != 'Heuristic':
        ranges.append(3)
    if args.threads == 'avg' and x_name != 'Threads' and y_name != 'Threads':
        ranges.append(4)

    for k in labels:
        key = [args.rays,args.vectorize,args.coherent,args.heuristic,args.threads]
        key[x_idx] = k
        acc = accumulate(key, y_idx, ranges)
        if acc != None:
            values.append(acc)

    labels = labels[0:len(values)]

    plt.clf()
    if x_name == 'Threads':
        labels = [float(l) for l in labels]
        plt.xscale('log')
        plt.xticks(labels)
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
        if y_name == 'Time':
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
            plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())
            
    plt.title('FCPQ Benchmark: {} vs {}'.format(x_name,y_name))
    plt.plot(labels, values)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    if args.save == '':
        plt.show()
    else:
        plt.savefig(os.path.join(args.save, '{}_{}__{}_{}_{}_{}_{}.png'.format(x_name,y_name,args.rays,args.vectorize,args.coherent,args.heuristic,args.threads)))


if args.x != 'auto' and args.y != 'auto':
    x_axis_idx = x_axis_types.index(args.x)
    y_axis_idx = y_axis_types.index(args.y)
    do_graph(x_axis_idx, y_axis_idx)

elif args.x != 'auto':
    x_axis_idx = x_axis_types.index(args.x)
    for i in range(len(y_axis_types)):
        do_graph(x_axis_idx, i)

elif args.y != 'auto':
    y_axis_idx = x_axis_types.index(args.y)
    for i in range(len(x_axis_types)):
        do_graph(i, y_axis_idx)

else:
    for i in range(len(x_axis_types)):
        for j in range(len(y_axis_types)):
            do_graph(i, j)
