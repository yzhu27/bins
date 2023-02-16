'''
---             ___                        __                     
---            /\_ \                      /\ \__                  
---      ___   \//\ \     __  __    ____  \ \ ,_\     __    _ __  
---     /'___\   \ \ \   /\ \/\ \  /',__\  \ \ \/   /'__`\ /\`'__\
---    /\ \__/    \_\ \_ \ \ \_\ \/\__, `\  \ \ \_ /\  __/ \ \ \/ 
---    \ \____\   /\____\ \ \____/\/\____/   \ \__\\ \____\ \ \_\ 
---     \/____/   \/____/  \/___/  \/___/     \/__/ \/____/  \/_/ 
                                                           
'''

import math
import re
import sys
import csv


the = {}
help = """
cluster.lua : an example csv reader script
(c)2022, Tim Menzies <timm@ieee.org>, BSD-2 

USAGE: cluster.lua  [OPTIONS] [-g ACTION]

OPTIONS:
  -F  --Far     distance to "faraway"  = .95
  -S  --Sample  sampling data size     = 512
  -d  --dump    on crash, dump stack   = false
  -f  --file    name of file           = ../etc/data/auto93.csv
  -g  --go      start-up action        = data
  -h  --help    show help              = false
  -m  --min     stop clusters at N^min = .5
  -p  --p       distance coefficient   = 2
  -s  --seed    random number seed     = 937162211

ACTIONS:
"""


#Summarize a stream of symbols.
class SYM:
    
    #no need to care about obj()
    
    ## line 35 function SYM.new(i)
    def __init__(self, at=0, txt=""):
        self.at = at
        self.txt = txt
        self.n = 0 # basic
        self.has = {} # similar as before?
        # dict for keeping data
        
        self.most = 0 #the frequency of the most frequent object
        self.mode = None #there is no mode initially

    # line 40 function SYM.add(i,x)
    def add(self, x):
        if x != "?":
            self.n +=  1
            
            #if x already exists in current record, just add frequency of its occurance
            #otherwise, create a new key and its new value-1            
            if x in self.has.keys():
                self.has[x] += 1
            else:
                self.has[x] = 1
            
            
            #after each insertion, check whether the frequency of new record becomes the most frequent one
            #by comparing with 'most'
            if self.has[x] > self.most:
                self.most = self.has[x]
                self.mode = x

    # line 47 function SYM.mid(i,x)
    def mid(self, *x):
        #here 'mid' stands for mode
        return self.mode

    # line 48 functon SYM.div(i,x,  fun, e)
    # fun() here should be an anonymous funciton
    #return the entropy
    def div(self, *x):
        e = 0
        for key in self.has:
            p = self.has[key] / self.n
            p = p*(math.log2(p))
            e += p
        
        return -e
    
    def rnd(self, x, *n):
        return x

    def dist(self, s1, s2):
        if s1 == '?' and s2 == '?':
            return 1
        elif s1 == s2:
            return 0
        else: return 1
    
#line 53
#Summarizes a stream of numbers.
class NUM:
    ## line 55 function NUM.new(i)
    def __init__(self, at=0, txt=""):
        self.at = at
        self.txt = txt
        self.n = 0 # basic
        
        self.mu = 0 # mean value of all
        self.m2 = 0 # standard deviation
        
        self.lo = math.inf # lowest value, initially set as MAX
        self.hi = -math.inf # highest value, initially set as MIN
        if txt=="":
            self.w = -1
        elif txt[-1]=="-":
            self.w = -1
        else: 
            self.w = 1
    # line 59 function NUM.add(i,x)
    # add `n`, update lo,hi and stuff needed for standard deviation
    def add(self, n):
        if n != "?":
            self.n +=  1
            
            d = n - self.mu
            
            self.mu += d/(self.n)
            self.m2 += d*(n - self.mu)
            
            self.lo = min(self.lo, n)
            self.hi = max(self.hi, n)

    # line 68 function NUM.mid(i,x)
    def mid(self, *x):
        #here 'mid' stands for mean
        return self.mu

    # line 69 functon NUM.div(i,x)
    # return standard deviation using Welford's algorithm
    def div(self, *x):
        if(self.m2 < 0 or self.n <2):
            return 0
        else:
            return pow((self.m2 / (self.n-1)), 0.5)
        
    def rnd(self, x, n): return x if x=="?" else rnd(x, n)

    def norm(self, n):
        if n == '?':
            return n
        else:
            return (n - self.lo) / (self.hi - self.lo)
    
    def dist(self, n1, n2):
        if n1 == '?' and n2 == '?': return 1
        n1 = self.norm(n1)
        n2 = self.norm(n2)
        if n1 == '?':
            n1 = (n2 < .5 and 1 or 0)
        if n2 == '?':
            n2 = (n1 < .5 and 1 or 0)
        return abs(n1 - n2)



class COLS:
    def __init__(self, names):
        self.names = names #dic
        self.all = {}
        self.klass = None
        self.x = {}
        self.y = {}
        
        for index, name in names.items():
            # all columns should be recorded in self.all, including those skipped columns
            # if the column starts with a capital character, it is Num
            # otherwise, it is Sym
            if name.istitle():
                curCol = push(self.all, NUM(index, name))
            else:
                curCol = push(self.all, SYM(index, name))    
            
            # lenOfName = len(name)
            
            # if a column ends with a ':', the column should be skipped and recorded nowhere except self.all
            
            # if there is any '+' or '-', the column should be regarded as a dependent variable
            # all dependent variables should be recoreded in self.y
            # on the contrary, those independent variables should be recorded in self.x
            if name[-1] != "X":
                if name [-1] == '!':
                    self.klass = curCol
                if "+" in name or "-" in name:
                    push(self.y, curCol)
                else:
                    push(self.x, curCol)
                
                # if a column name ends with a '!', this column should be recorded AS self.klass
                # NOTICE THAT IT IS "AS", NOT "INCLUDED IN"

    def add(self, row):
        for _,t in self.y.items():
            t.add(row.cells[t.at])

                
        for _,t in self.x.items():
            t.add(row.cells[t.at])


class ROW:
    def __init__(self, t):
        self.cells = t

class DATA:
    def __init__(self , src):
        self.rows = {}
        self.cols = None
        def fun(x):
            self.add(x)
        if type(src) == str:
            Csv(src , fun)
        else:
            if src:
                #map(src , fun)
                self.add(src)
            else:
                map({} , fun)
    
    def add(self , t):
        if self.cols:
            t = t if type(t) == ROW else ROW(t)
            push(self.rows , t)
            self.cols.add(t) #COLS.add()
        else:
            self.cols = COLS(t)
    
    def clone(self , init):
        data = DATA(self.cols.names)
        def fun(x):
            data.add(x)
        map(init or {} , fun)
        return data
            
    def stats(self , what , cols , nPlaces):
        def fun(k , col):
            if what == 'div':
                return col.rnd(col.div(col) , nPlaces)
            else:
                return col.rnd(col.mid(col) , nPlaces)
        u = {}
        for i in range(len(cols)):
            k = cols[i].txt
            u[k] = fun(k , cols[i])
        res = {}
        for k in sorted(u.keys()):
            res[k] = u[k]
        return res
    
    def better(self , row1 , row2):
        s1 = 0
        s2 = 0
        ys = self.cols.y
        for _ , col in ys.items():
            x = col.norm(row1.cells[col.at])
            y = col.norm(row2.cells[col.at])
            s1 -= math.exp(col.w * (x - y) / len(ys))
            s2 -= math.exp(col.w * (y - x) / len(ys))
        return (s1 / len(ys)) < (s2 / len(ys))

    
    def dist(self , row1 , row2 , *cols):
        n , d = 0 , 0
        if cols is None: cols = self.cols.x
        for _ , col in self.cols.x.items():
            n += 1
            d += col.dist(row1.cells[col.at] , row2.cells[col.at]) ** the['p']
        return (d / n) ** (1 / the['p']) 

    def around(self , row1 , rows=None , cols=None): #--> list[dict{row: , dist: }]
        def fun(row2):
            dic = {}
            dic['row'] = row2
            dic['dist'] = self.dist(row1 , row2 , cols)
            return dic
        tmp = map(rows or self.rows , fun) #dic{dic{}}
        tmp = list(tmp.values()) # [dict]
        return sort(tmp , lt('dist')) 
    
    def half(self , **kwargs):
        def dist(row1 , row2):
            return self.dist(row1 , row2 , kwargs['cols'] if 'cols' in kwargs else None)

        def project(row):
            dic = {}
            dic['row'] = row
            dic['dist'] = cosine(dist(row , A) , dist(row , B) , c)
            return dic
        
        rows = kwargs['rows'] if 'rows' in kwargs else self.rows
        some = many(rows , the['Sample'])
        A = kwargs['above'] if ('above' in kwargs and kwargs['above']) else any(some)
        B = self.around(row1=A , rows=some)[int(the['Far'] * len(rows)) // 1]['row']
        c = dist(A , B)
        left , right = {} , {}
        #print(sort(list(map(rows , project).values()) , lt('dist')))
        for n , tmp in enumerate(sort(list(map(rows , project).values()) , lt('dist')) , 1):
            if n <= len(rows) // 2:
                push(left , tmp['row'])
                mid = tmp['row']
            else:
                push(right , tmp['row'])
        return left , right , A , B , mid , c
        
    
    def cluster(self , **kwargs):
        rows = kwargs['rows'] if 'rows' in kwargs else self.rows
        min = kwargs['min'] if 'min' in kwargs else len(rows) ** the['min']
        cols = kwargs['cols'] if 'cols' in kwargs else self.cols.x
        node = {}
        node['data'] = self.clone(rows)
        if len(rows) > 2 * min:
            left , right , node['A'] , node['B'] , node['mid'], _ = self.half(rows=rows , cols=cols , above=kwargs['above'] if 'above' in kwargs else None)
            node['left'] = self.cluster(rows=left , min=min , cols=cols , above=node['A'])
            node['right'] = self.cluster(rows=right , min=min , cols=cols , above=node['B'])
        return node
    
    def sway(self , **kwargs):
        rows = kwargs['rows'] if 'rows' in kwargs else self.rows
        min = kwargs['min'] if 'min' in kwargs else len(rows) ** the['min']
        cols = kwargs['cols'] if 'cols' in kwargs else self.cols.x
        node = {}
        node['data'] = self.clone(rows)
        if len(rows) > 2 * min:
            left , right , node['A'] , node['B'] , node['mid'], _ = self.half(rows=rows , cols=cols , above=kwargs['above'] if 'above' in kwargs else None)
            if self.better(node['B'] , node['A']):
                left , right , node['A'] , node['B'] = right , left , node['B'] , node['A']
            node['left'] = self.sway(rows=left , min=min , cols=cols , above=node['A'])
        return node

       
## Misc

def show(node, what, cols, nPlaces, lvl:int=None):
    if node:
        lvl = lvl if lvl is not None else 0
        res = '| ' * lvl + str(len(node['data'].rows)) + '  '
        if 'left' not in node or lvl == 0:
            print(res + o(node['data'].stats("mid",node['data'].cols.y,nPlaces)))
        else:
            print(res)
        if 'left' in node:
            show(node['left'], what, cols, nPlaces, lvl+1)
        if 'right' in node:
            show(node['right'], what, cols, nPlaces, lvl+1)


## Numerics 

Seed = 937162211

# n ; a integer lo..hi-1
def rint(lo, hi):
    return math.floor(0.5 + rand(lo, hi))

# n; a float "x" lo<=x < x
def rand(lo, hi):
    global Seed
    lo = lo or 0
    hi = hi or 1
    Seed = (16807 * Seed) % 2147483647
    return lo + (hi-lo) * Seed / 2147483647

# num. return `n` rounded to `nPlaces`
def rnd(n, nPlaces=3):
    mult = 10**nPlaces
    return math.floor(n * mult + 0.5) / mult

# n,n;  find x,y from a line connecting `a` to `b`
def cosine(a, b, c):
    x1 = (a**2 + c**2 - b**2) / (2*c)
    x2 = max(0, min(1, x1))
    y = math.sqrt(abs(a**2 - x2**2))
    return x2, y

## Lists

# Note the following conventions for `map`.
# - If a nil first argument is returned, that means :skip this result"
# - If a nil second argument is returned, that means place the result as position size+1 in output.
# - Else, the second argument is the key where we store function output.

# t; map a function `fun`(v) over list (skip nil results)
def map(t:dict, fun):
    u = {}
    for k, v in t.items():
        u[k]=fun(v)
    return u

# t; map function `fun`(k,v) over list (skip nil results)
def kap(t:dict, fun):
    u = {}
    for k, v in t.items():
        u[k]=fun(k, v)
    return u

# t; return `t`,  sorted by `fun` (default= `<`)
def sort(t:list, fun = lambda x: x.keys()):
    return sorted(t, key=fun)

def lt(x: str):
    def fun(dic):
        return dic[x]
    return fun

# ss; return list of table keys, sorted
def keys(t:list):
    return sorted(kap(t, lambda k, _:k))

# any; push `x` to end of list; return `x` 
def push(t:dict, x):
    t[len(t)] = x
    return x

# x; returns one items at random
def any(t):
    return list(t.values())[rint(len(t),1)-1]

# t1; returns some items from `t`
def many(t, n):
    u = {}
    for i in range(0, n):
        u[i] = any(t)
    return u
## Strings



def fmt(sControl , *elements): # emulate printf
    return (sControl%(elements)) 
#test
##a=1
##b=2
##print(fmt("%s and %s" , a , b)) #--> "1 and 2"


def o(t , *isKeys): #--> s; convert `t` to a string. sort named keys.
    if type(t) != dict:
        return str(t)
    
    def fun(k , v):
        if not re.findall('[^_]' , str(k)):
            return fmt(":%s %s",o(k),o(v))
    
    if len(t) > 0 and not isKeys:
        tmp = map(t , o)
    else:
        tmp = sort(kap(t , fun))

    def concat(tmp:dict):
        res = []
        for k , v in tmp.items():
            res.append(':' + str(k))
            res.append(v)
        return res
    return '{' + ' '.join(concat(tmp)) + '}'

def oo(t):
    print(o(t))
    return t

def coerce(s):
    def fun(s1):
        if s1 == 'true':
            return True
        if s1 == 'false':
            return False 
        return s1.strip()
    if s.isdigit():
        return int(s)
    try:
        tmp = float(s)
        return tmp
    except ValueError:
        return fun(s)
    

def Csv(fname, fun):
    n=0
    with open(fname,'r') as src:
        rdr = csv.reader(src, delimiter=',')
        for l in rdr:
            d={}
            for v in l:
                d[len(d)]=coerce(v)
            n+=len(d)
            fun(d)
    return n

### Main


def settings(s):  # --> t;  parse help string to extract a table of options
    t = {}
    # match the contents like: '-d  --dump  on crash, dump stack = false'
    res = r"[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)"
    m = re.findall(res, s)
    for key, value in m:
        t[key] = coerce(value)
    return t
# test
# print(settings(help)) --> {'dump': False, 'go': 'data', 'help': False, 'seed': 937162211}

# Update settings from values on command-line flags. Booleans need no values


def cli(t, list):
    slots = list[1:]
    # search the key and value we want to update
    for slot, v in t.items():
        # give each imput slot an index(begin from 0)
        for n, x in enumerate(slots):
            # match imput slot with the.keys: x == '-e' or '--eg'
            if x == ('-'+slot[0]) or x == ('--'+slot):
                v = str(v)
                # we just flip the defeaults
                if v == 'True':
                    v = 'false'
                elif v == 'False':
                    v = 'true'
                else:
                    v = slots[n+1]
                t[slot] = coerce(v)
    return t



def main(options, help, funs, *k):
    saved = {}
    fails = 0
    for k, v in cli(settings(help), sys.argv).items():
        options[k] = v
        saved[k] = v
    if options['help']:
        print(help)
    

    else:
        for what, fun in funs.items():
            if options['go'] == 'all' or what == options['go']:
                for k, v in saved.items():
                    options[k] = v
                if fun() == False:
                    fails += 1
                    print("❌ fail:", what)
                else:
                    print("✅ pass:", what)


## Examples

egs = {}
def eg(key, str, fun):  #--> nil; register an example.
    global help
    egs[key] = fun
    #help = help + f'  -g  {key}\t{str}\n'
    help = help + fmt('  -g  %s\t%s\n', key, str)




if __name__=='__main__':

    # eg("crash","show crashing behavior", function()
    #   return the.some.missing.nested.field end)
    def thefun():
        global the
        return oo(the)
    eg("the","show settings", thefun)

    def symfun():
        sym = SYM()
        for x in ["a","a","a","a","b","b","c"]:
            sym.add(x)
        return "a" == sym.mid() and 1.379 == rnd(sym.div())
    eg("sym","check syms", symfun)

    def numfun():
        num = NUM()
        for x in [1,1,1,1,2,2,3]:
            num.add(x)
        return 11/7 == num.mid() and 0.787 == rnd(num.div())
    eg("num", "check nums", numfun)
    
    def datafun():
        data = DATA(the["file"])
        return len(data.rows) == 398 and\
               data.cols.y[0].w == -1 and\
               data.cols.x[0].at == 0 and\
               len(data.cols.x) == 4   # second & third line should be [0]s 
                                       # instead of the [1]s in lua and third
                                       # line should be 0 because python
                                       # is 0 index and lua is 1 index
    eg("data","read DATA csv", datafun)

    def clonefun():
        data1 = DATA(the["file"])
        data2 = data1.clone(data1.rows)
        return len(data1.rows) == len(data2.rows) and\
               data1.cols.y[0].w == data2.cols.y[0].w and\
               data1.cols.x[0].at == data2.cols.x[0].at and\
               len(data1.cols.x) == len(data2.cols.x)
    eg("clone", "duplicate structure", clonefun)

    def aroundfun():
        data = DATA(the["file"])
        tmp = []
        for num in data.rows[0].cells.values():
            tmp.append(str(num))
        print(str(0)+"  "+str(0)+"  "+ '{' + ' '.join(tmp) + '}')
        for n, t in enumerate(data.around(data.rows[0])):
            if (n+1) % 50 == 0:
                if n == 0:
                    continue
                tmp = []
                for num in t['row'].cells.values():
                    tmp.append(str(num))
                print(str(n+1)+"  "+str(rnd(t['dist'], 2))+" "+ '{' + ' '.join(tmp) + '}')
    eg("around", "sorting nearest neighbors", aroundfun)

    def halffun():
        data = DATA(the["file"])
        left, right, A, B, mid, c = data.half()
        print(str(len(left))+"   "+str(len(right))+"   "+str(len(data.rows)))
        tmpA = []
        for num in A.cells.values():
            tmpA.append(str(num))
        print('{' + ' '.join(tmpA) + '}     '+str(c))
        tmpmid = []
        for num in mid.cells.values():
            tmpmid.append(str(num))
        print('{' + ' '.join(tmpmid) + '}')
        tmpB = []
        for num in B.cells.values():
            tmpB.append(str(num))
        print('{' + ' '.join(tmpB) + '}')
    eg("half", "1-level bi-clustering", halffun)

    def clusterfun():
        data = DATA(the["file"])
        show(data.cluster(), 'mid', data.cols.y, 1)
        return True
    eg("cluster", "N-level bi-clustering", clusterfun)

    def optimizefun():
        data = DATA(the["file"])
        show(data.sway(), 'mid', data.cols.y, 1)
        return True
    eg("optimize", "semi-supervised optimization", optimizefun)


    main(the, help, egs)