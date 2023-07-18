#! /users12/yptong/anaconda3/bin/python
import sys

if len(sys.argv) != 3:
    print('Wrong Arguments!!!')

else:
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')[:-1]
        lines = [line for line in lines if line.startswith('[INFO') or line.startswith('{') or line.startswith('A')
                                            or 'pola' in line or 'itst' in line ]
    with open(sys.argv[2], 'w', encoding='utf-8') as f2:
        f2.write('\n'.join(lines))