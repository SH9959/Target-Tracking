from math import sqrt
from sys import argv
from typing import Tuple
from functools import reduce
def pick_pos(line:str):
    [f,x,y] = line.split(" ")
    return (int(float(x)),int(float(y)))

def read_input(filename):
    with open(filename,'r') as f:
        data = f.readlines()
        return [pick_pos(line) for line in data if not (len(line)==0)]
    pass

Data = read_input(argv[1])
Criteria = read_input(argv[2])

P2 = Tuple[int,int]
PosList = list[P2]
def pos_diff_root(ps : Tuple[P2,P2]):
        p1,p2 = ps
        x1,y1 = p1;x2,y2=p2
        return sqrt((x1-x2)**2 + (y1-y2)**2)
        pass

def mean_square(data:PosList,criteria:PosList):
    return sum(map(pos_diff_root,zip(data,criteria))) / len(data)

total_error = mean_square(Data,Criteria)

print(f"图片共 {len(Data)} 帧") 
print (f"{argv[1]} 相对于 {argv[2]} 的均方根误差是 {total_error} (像素平方)")
