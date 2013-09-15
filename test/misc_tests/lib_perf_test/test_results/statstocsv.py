#! /usr/bin/env python

# Imports ---

import re, sys, os

# Constants ---

data =  \
"""Contiguous write access 800^3
 - Dense           | alloc: 434       | run: 380       | total: 814       | mem: 2048000744     | rss: 2048204800     | check sum: 512000000
 - Sparse 8        | alloc: 3         | run: 5174      | total: 5177      | mem: 2064000400     | rss: 2080235520     | check sum: 512000000
 - VDB 8           | alloc: 0         | run: 4904      | total: 4904      | mem: 2143866376     | rss: 2172227584     | check sum: 512000000
 - Sparse 16       | alloc: 0         | run: 6339      | total: 6339      | mem: 2050000400     | rss: 2052349952     | check sum: 512000000
 - VDB 16          | alloc: 0         | run: 5623      | total: 5623      | mem: 2116935060     | rss: 2120822784     | check sum: 512000000
 - Sparse 32       | alloc: 0         | run: 4594      | total: 4594      | mem: 2048250400     | rss: 2048884736     | check sum: 512000000
 - VDB 32          | alloc: 0         | run: 3945      | total: 3945      | mem: 2112854536     | rss: 2113699840     | check sum: 512000000
Contiguous write access (preallocated) 800^3
 - Dense           | alloc: 0         | run: 193       | total: 193       | mem: 2048000744     | rss: 2048819200     | check sum: 512000000
 - Sparse 8        | alloc: 0         | run: 6127      | total: 6127      | mem: 2064000400     | rss: 2080415744     | check sum: 512000000
 - VDB 8           | alloc: 0         | run: 5506      | total: 5506      | mem: 2143866376     | rss: 2172313600     | check sum: 512000000
 - Sparse 16       | alloc: 0         | run: 6102      | total: 6102      | mem: 2050000400     | rss: 2052435968     | check sum: 512000000
 - VDB 16          | alloc: 0         | run: 5198      | total: 5198      | mem: 2116935060     | rss: 2120896512     | check sum: 512000000
 - Sparse 32       | alloc: 0         | run: 3875      | total: 3875      | mem: 2048250400     | rss: 2048958464     | check sum: 512000000
 - VDB 32          | alloc: 0         | run: 3141      | total: 3141      | mem: 2112854536     | rss: 2113769472     | check sum: 512000000
"""

# Script ---

input = open(sys.argv[1]).readlines()

output = open(sys.argv[1][:-3] + "csv", "w")

for line in input:
    x = line.split("|")
    if len(x) == 1:
        print x[0]
        output.writelines([x[0]])
    if len(x) == 7:
        items = [x[0].strip()[2:], 
                 x[1].split()[1],
                 x[2].split()[1],
                 x[3].split()[1],
                 x[4].split()[1],
                 x[5].split()[1]]
        print ", ".join(items)
        output.writelines([",".join(items) + "\n"])
        
