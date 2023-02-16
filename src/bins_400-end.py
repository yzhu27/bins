
import math
import re
import sys
import csv


the = {}
help = """
bins: multi-objective semi-supervised discetization
(c) 2023 Tim Menzies <timm@ieee.org> BSD-2
  
USAGE: lua bins.lua [OPTIONS] [-g ACTIONS]
  
OPTIONS:
  -b  --bins    initial number of bins       = 16
  -c  --cliffs  cliff's delta threshold      = .147
  -f  --file    data file                    = ../etc/data/auto93.csv
  -F  --Far     distance to distant          = .95
  -g  --go      start-up action              = nothing
  -h  --help    show help                    = false
  -H  --Halves  search space for clustering  = 512
  -m  --min     size of smallest cluster     = .5
  -M  --Max     numbers                      = 512
  -p  --p       dist coefficient             = 2
  -r  --rest    how many of rest to sample   = 4
  -R  --Reuse   child splits reuse a parent pole = true
  -s  --seed    random number seed           = 937162211
"""


# Summarize a stream of symbols.
class SYM:

    # no need to care about obj()

    # line 35 function SYM.new(i)
    def __init__(self, at=0, txt=""):
        self.at = at
        self.txt = txt
        self.n = 0  # basic
        self.has = {}  # similar as before?
        # dict for keeping data

        self.most = 0  # the frequency of the most frequent object
        self.mode = None  # there is no mode initially

    # line 40 function SYM.add(i,x)
    def add(self, x):
        if x != "?":
            self.n += 1

            # if x already exists in current record, just add frequency of its occurance
            # otherwise, create a new key and its new value-1
            if x in self.has.keys():
                self.has[x] += 1
            else:
                self.has[x] = 1

            # after each insertion, check whether the frequency of new record becomes the most frequent one
            # by comparing with 'most'
            if self.has[x] > self.most:
                self.most = self.has[x]
                self.mode = x

    # line 47 function SYM.mid(i,x)
    def mid(self, *x):
        # here 'mid' stands for mode
        return self.mode

    # line 48 functon SYM.div(i,x,  fun, e)
    # fun() here should be an anonymous funciton
    # return the entropy
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
        else:
            return 1

# line 53
# Summarizes a stream of numbers.


class NUM:
    # line 55 function NUM.new(i)
    def __init__(self, at=0, txt=""):
        self.at = at
        self.txt = txt
        self.n = 0  # basic

        self.mu = 0  # mean value of all
        self.m2 = 0  # standard deviation

        self.lo = math.inf  # lowest value, initially set as MAX
        self.hi = -math.inf  # highest value, initially set as MIN
        if txt == "":
            self.w = -1
        elif txt[-1] == "-":
            self.w = -1
        else:
            self.w = 1
    # line 59 function NUM.add(i,x)
    # add `n`, update lo,hi and stuff needed for standard deviation

    def add(self, n):
        if n != "?":
            self.n += 1

            d = n - self.mu

            self.mu += d/(self.n)
            self.m2 += d*(n - self.mu)

            self.lo = min(self.lo, n)
            self.hi = max(self.hi, n)

    # line 68 function NUM.mid(i,x)
    def mid(self, *x):
        # here 'mid' stands for mean
        return self.mu

    # line 69 functon NUM.div(i,x)
    # return standard deviation using Welford's algorithm
    def div(self, *x):
        if (self.m2 < 0 or self.n < 2):
            return 0
        else:
            return pow((self.m2 / (self.n-1)), 0.5)

    def rnd(self, x, n): return x if x == "?" else rnd(x, n)

    def norm(self, n):
        if n == '?':
            return n
        else:
            return (n - self.lo) / (self.hi - self.lo)

    def dist(self, n1, n2):
        if n1 == '?' and n2 == '?':
            return 1
        n1 = self.norm(n1)
        n2 = self.norm(n2)
        if n1 == '?':
            n1 = (n2 < .5 and 1 or 0)
        if n2 == '?':
            n2 = (n1 < .5 and 1 or 0)
        return abs(n1 - n2)


class COLS:
    def __init__(self, names):
        self.names = names  # dic
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
                if name[-1] == '!':
                    self.klass = curCol
                if "+" in name or "-" in name:
                    push(self.y, curCol)
                else:
                    push(self.x, curCol)

                # if a column name ends with a '!', this column should be recorded AS self.klass
                # NOTICE THAT IT IS "AS", NOT "INCLUDED IN"

    def add(self, row):
        for _, t in self.y.items():
            t.add(row.cells[t.at])

        for _, t in self.x.items():
            t.add(row.cells[t.at])


class ROW:
    def __init__(self, t):
        self.cells = t


class DATA:
    def __init__(self, src):
        self.rows = {}
        self.cols = None

        def fun(x):
            self.add(x)
        if type(src) == str:
            Csv(src, fun)
        else:
            if src:
                # map(src , fun)
                self.add(src)
            else:
                map({}, fun)

    def add(self, t):
        if self.cols:
            t = t if type(t) == ROW else ROW(t)
            push(self.rows, t)
            self.cols.add(t)  # COLS.add()
        else:
            self.cols = COLS(t)

    def clone(self, init):
        data = DATA(self.cols.names)

        def fun(x):
            data.add(x)
        map(init or {}, fun)
        return data

    def stats(self, what, cols, nPlaces):
        def fun(k, col):
            if what == 'div':
                return col.rnd(col.div(col), nPlaces)
            else:
                return col.rnd(col.mid(col), nPlaces)
        u = {}
        for i in range(len(cols)):
            k = cols[i].txt
            u[k] = fun(k, cols[i])
        res = {}
        for k in sorted(u.keys()):
            res[k] = u[k]
        return res

    def better(self, row1, row2):
        s1 = 0
        s2 = 0
        ys = self.cols.y
        for _, col in ys.items():
            x = col.norm(row1.cells[col.at])
            y = col.norm(row2.cells[col.at])
            s1 -= math.exp(col.w * (x - y) / len(ys))
            s2 -= math.exp(col.w * (y - x) / len(ys))
        return (s1 / len(ys)) < (s2 / len(ys))

    def dist(self, row1, row2, *cols):
        n, d = 0, 0
        if cols is None:
            cols = self.cols.x
        for _, col in self.cols.x.items():
            n += 1
            d += col.dist(row1.cells[col.at], row2.cells[col.at]) ** the['p']
        return (d / n) ** (1 / the['p'])

    # --> list[dict{row: , dist: }]
    def around(self, row1, rows=None, cols=None):
        def fun(row2):
            dic = {}
            dic['row'] = row2
            dic['dist'] = self.dist(row1, row2, cols)
            return dic
        tmp = map(rows or self.rows, fun)  # dic{dic{}}
        tmp = list(tmp.values())  # [dict]
        return sort(tmp, lt('dist'))

    def half(self, **kwargs):
        def dist(row1, row2):
            return self.dist(row1, row2, kwargs['cols'] if 'cols' in kwargs else None)

        def project(row):
            dic = {}
            dic['row'] = row
            dic['dist'] = cosine(dist(row, A), dist(row, B), c)
            return dic

        rows = kwargs['rows'] if 'rows' in kwargs else self.rows
        some = many(rows, the['Sample'])
        A = kwargs['above'] if (
            'above' in kwargs and kwargs['above']) else any(some)
        B = self.around(row1=A, rows=some)[
            int(the['Far'] * len(rows)) // 1]['row']
        c = dist(A, B)
        left, right = {}, {}
        # print(sort(list(map(rows , project).values()) , lt('dist')))
        for n, tmp in enumerate(sort(list(map(rows, project).values()), lt('dist')), 1):
            if n <= len(rows) // 2:
                push(left, tmp['row'])
                mid = tmp['row']
            else:
                push(right, tmp['row'])
        return left, right, A, B, mid, c

    def cluster(self, **kwargs):
        rows = kwargs['rows'] if 'rows' in kwargs else self.rows
        min = kwargs['min'] if 'min' in kwargs else len(rows) ** the['min']
        cols = kwargs['cols'] if 'cols' in kwargs else self.cols.x
        node = {}
        node['data'] = self.clone(rows)
        if len(rows) > 2 * min:
            left, right, node['A'], node['B'], node['mid'], _ = self.half(
                rows=rows, cols=cols, above=kwargs['above'] if 'above' in kwargs else None)
            node['left'] = self.cluster(
                rows=left, min=min, cols=cols, above=node['A'])
            node['right'] = self.cluster(
                rows=right, min=min, cols=cols, above=node['B'])
        return node

    def sway(self, **kwargs):
        rows = kwargs['rows'] if 'rows' in kwargs else self.rows
        min = kwargs['min'] if 'min' in kwargs else len(rows) ** the['min']
        cols = kwargs['cols'] if 'cols' in kwargs else self.cols.x
        node = {}
        node['data'] = self.clone(rows)
        if len(rows) > 2 * min:
            left, right, node['A'], node['B'], node['mid'], _ = self.half(
                rows=rows, cols=cols, above=kwargs['above'] if 'above' in kwargs else None)
            if self.better(node['B'], node['A']):
                left, right, node['A'], node['B'] = right, left, node['B'], node['A']
            node['left'] = self.sway(
                rows=left, min=min, cols=cols, above=node['A'])
        return node


# Misc

def show(node, what, cols, nPlaces, lvl: int = None):
    if node:
        lvl = lvl if lvl is not None else 0
        res = '| ' * lvl + str(len(node['data'].rows)) + '  '
        if 'left' not in node or lvl == 0:
            print(res + o(node['data'].stats("mid",
                  node['data'].cols.y, nPlaces)))
        else:
            print(res)
        if 'left' in node:
            show(node['left'], what, cols, nPlaces, lvl+1)
        if 'right' in node:
            show(node['right'], what, cols, nPlaces, lvl+1)


# Numerics

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

# Lists

# Note the following conventions for `map`.
# - If a nil first argument is returned, that means :skip this result"
# - If a nil second argument is returned, that means place the result as position size+1 in output.
# - Else, the second argument is the key where we store function output.

# t; map a function `fun`(v) over list (skip nil results)


def map(t: dict, fun):
    u = {}
    for k, v in t.items():
        u[k] = fun(v)
    return u

# t; map function `fun`(k,v) over list (skip nil results)


def kap(t: dict, fun):
    u = {}
    for k, v in t.items():
        u[k] = fun(k, v)
    return u

# t; return `t`,  sorted by `fun` (default= `<`)


def sort(t: list, fun=lambda x: x.keys()):
    return sorted(t, key=fun)


def lt(x: str):
    def fun(dic):
        return dic[x]
    return fun

# ss; return list of table keys, sorted


def keys(t: list):
    return sorted(kap(t, lambda k, _: k))

# any; push `x` to end of list; return `x`


def push(t: dict, x):
    t[len(t)] = x
    return x

# x; returns one items at random


def any(t):
    return list(t.values())[rint(len(t), 1)-1]

# t1; returns some items from `t`


def many(t, n):
    u = {}
    for i in range(0, n):
        u[i] = any(t)
    return u
# Strings


def fmt(sControl, *elements):  # emulate printf
    return (sControl % (elements))
# test
# a=1
# b=2
# print(fmt("%s and %s" , a , b)) #--> "1 and 2"

# Return a portion of `t`; go,stop,inc defaults to 1,#t,1.
# Negative indexes are supported.
def slice(t, go, stop ,inc):
    if go and go < 0 : go += len(t)
    if stop and stop < 0 : stop += len(t)
    u = {}
    for j in range(int(go or 0),int(stop or len(t)),int(inc or 1)):
        u[len(u)].append(t[j])
    return u
    

def o(t, *isKeys):  # --> s; convert `t` to a string. sort named keys.
    if type(t) != dict:
        return str(t)

    def fun(k, v):
        if not re.findall('[^_]', str(k)):
            return fmt(":%s %s", o(k), o(v))

    if len(t) > 0 and not isKeys:
        tmp = map(t, o)
    else:
        tmp = sort(kap(t, fun))

    def concat(tmp: dict):
        res = []
        for k, v in tmp.items():
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
    n = 0
    with open(fname, 'r') as src:
        rdr = csv.reader(src, delimiter=',')
        for l in rdr:
            d = {}
            for v in l:
                d[len(d)] = coerce(v)
            n += len(d)
            fun(d)
    return n

# Main


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
    y,n=0,0
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


# Examples

egs = {}


def go(key, str, fun):  # --> nil; register an example.
    global help
    egs[key] = fun
    # help = help + f'  -g  {key}\t{str}\n'
    help = help + fmt('  -g  %s\t%s\n', key, str)

b4 = {}
if __name__ == '__main__':

    # eg("crash","show crashing behavior", function()
    #   return the.some.missing.nested.field end)
    def thefun():
        global the
        return oo(the)
    go("the", "show options", thefun)

    def randfun():
        global Seed
        Seed = 1
        t = {}
        for _ in range(0, 1000):
            push(t, rint(100))
        Seed = 1
        u = {}
        for _ in range(0, 1000):
            push(u, rint(100))
        for k, v in enumerate(t):
            assert (v == u[k])
    go("rand", "demo random number generation", randfun)

    def somefun():
        global the
        the['Max'] = 32
        num1 = NUM()
        for i in range(0, 10000):
            add(num1, i)
        oo(has(num1))
    go("some", "demo of reservoir sampling", somefun)

    def symsfun():
        sym = SYM()
        for x in ["a", "a", "a", "a", "b", "b", "c"]:
            sym.add(x)
        return 1.379 == rnd(div(sym))
    go("syms","demo SYMS", symsfun)

    def numsfun():
        num1, num2 = NUM(), NUM()
        for _ in range(0, 10000):
            add(num1, rand())
        for _ in range(0, 10000):
            add(num2, rand()**2)
        print("1  "+str(rnd(mid(num1)))+str(rnd(div(num1))))
        print("2  "+str(rnd(mid(num2)))+str(rnd(div(num2))))
        return .5 == rnd(mid(num1)) and mid(num1) > mid(num2)
    go("nums", "demo of NUM", numsfun)

    def csvfun():
        n = 0
        def tmp(t):
            return len(t)
        n = Csv(the["file"], tmp)
        return n==8*399
    go("csv","reading csv files", csvfun)
    
    
    def datafun():
        data = DATA(the["file"])
        col = data.cols.x[0]
        print(str(col.lo)+" "+str(col.hi)+" "+str(mid(col))+" "+div(col))
        oo(stats(data))
    go("data", "showing data sets", datafun)

    

    def clonefun():
        data1 = DATA(the["file"])
        data2 = data1.clone(data1.rows)
        oo(stats(data1))
        oo(stats(data2))
    go("clone","replicate structure of a DATA", clonefun)



    def cliffsfun():
        assert(False == cliffsDelta({8,7,6,2,5,8,7,3},{8,7,6,2,5,8,7,3}),"1")
        assert(True == cliffsDelta({8,7,6,2,5,8,7,3}, {9,9,7,8,10,9,6}),"2")
        t1,t2 = {},{}
        for _ in range(0,1000): push(t1, rand())
        for _ in range(0,1000): push(t2, math.sqrt(rand()))
        assert(False == cliffsDelta(t1,t1), "3")
        assert(True == cliffsDelta(t1,t2), "4") 
        diff = False
        j = 1.0
        while(not diff):
            t3 = map(t1,lambda x:x*j)
            diff = cliffDelta(t1,t3)
            print("> "+str(rnd(j))+"  "+str(diff))
            j=j*1.025
    go("cliffs","stats tests", cliffsfun)

    def distfun():
        data = DATA(the["file"])
        num = NUM()
        for _,row in enumerate(data.rows):
            add(num,dist(data,row,data.rows[0]))
        oo({'lo':num.lo,'hi':num.hi,'mid':rnd(mid(num)),'div':rnd(div(num))})
    go("dist","distance test", distfun)

    def halffun():
        data = DATA(the["file"])
        left, right, A, B, c = data.half()
        print(str(len(left))+"   "+str(len(right))+"   "+str(len(data.rows)))
        tmpL = []
        for num in left.cells.values():
            tmpL.append(str(num))
        print('{' + ' '.join(tmpL) + '}     '+str(c))
        tmpR = []
        for num in right.cells.values():
            tmpR.append(str(num))
        print('{' + ' '.join(tmpR) + '}')
    go("half","divide data in half", halffun)

    def treefun():
        showTree(tree(DATA(the['file'])))
    go("tree","make snd show tree of clusters", treefun)

    def swayfun():
        data = DATA.read(the['file'])
        best,rest = sway(data)
        print("\nall    "+oo(stats(data)))
        print("       "+oo(stats(data,div)))
        print("\nbest    "+oo(stats(best)))
        print("       "+oo(stats(best,div)))
        print("\nrest    "+oo(stats(rest)))
        print("       "+oo(stats(rest,div)))
        print("\nall != best?   "+oo(diffs(best.cols.y, data.cols.y)))
        print("best != rest?   "+oo(diffs(best.cols.y, rest.cols.y)))
    go("sway","optimizing", swayfun)


    def binsfun():
        global b4
        data = DATA.read(the['file'])
        best,rest = sway(data)
        print("all     "+oo({'best':len(best.rows), 'rest':len(rest.rows)}))
        for k,t in enumerate(bins(data.cols.x,{'best':best.rows,'rest':rest.rows})):
            for _,range in t.items():
                if range.txt != b4: print("  ")
                b4 = range.txt
                print(str(range.txt)+"  "+str(range.lo)+"  "+str(range.hi)+"  "+rnd(value(range.y.has,len(best.rows,len(rest.rows),"best")))+"  "+oo(range.y.has))
        show(data.sway(), 'mid', data.cols.y, 1)
        return True
    go("bins", "find deltas between best and rest", binsfun)

    main(the, help, egs)
