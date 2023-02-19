
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

# Create a `SYM` to summarize a stream of symbols.
def SYM(n=0, s=""):
    return {"at": n, 
            "txt": s, 
            "n": 0, 
            "mode": None, 
            "most": 0, 
            "isSym": True, 
            "has": {}
    }

# Create a `NUM` to summarize a stream of numbers.
def NUM(n=0, s=""):
    return {"at": n, 
            "txt": s, 
            "n": 0, 
            "hi": -math.inf, 
            "lo": math.inf, 
            "ok": True, 
            "has": {}, 
            "w": -1 if (s or "").endswith("-") else 1}

# Create a `COL` to summarize a stream of data for a column.
def COL(n, s):
    col = NUM(n, s) if s.startswith(("[A-Z]")) else SYM(n, s)
    col["isIgnored"] = col["txt"].endswith("X")
    col["isKlass"] = col["txt"].endswith("!")
    
    col["isGoal"] = col['txt'].endswith(('!', '+', '-'))
    return col

# Create a set of `NUM`s or `SYM`s columns.
def COLS(ss):
    cols = {"names": ss, "all": {}, "x": {}, "y": {}}
    for n, s in enumerate(ss):
        # use push() here, defined in later code
        # col = cols['all'].update(COL(n, s))
        col = push(cols['all'], COL(n,s))
        
        if not col['isIgnored']:
            if col['isKlass']:
                cols['klass'] = col
            if col['isGoal']:
                # cols['y'].update(col)
                push(cols['y'], col)
            else:
                # cols['x'].update(col)
                push(cols['x'], col)
    return cols

# Create a RANGE  that tracks the y dependent values seen in 
# the range `lo` to `hi` some independent variable in column number `at` whose name is `txt`. 
# Note that the way this is used (in the `bins` function, below)
# for  symbolic columns, `lo` is always the same as `hi`.
def RANGE(at, txt, lo, hi=None):
    return {"at": at, 
            "txt": txt, 
            "lo": lo, 
            "hi": hi if lo is None else lo, 
            "y": SYM()}

# Create a `DATA` to contain `rows`, summarized in `cols`.
class DATA:
    def __init__(self):
        self.rows = {}
        self.cols = None

    # Create a new DATA by reading csv file whose first row 
    # are the comma-separate names processed by `COLS` (above).
    # into a new `DATA`. Every other row is stored in the DATA by
    # calling the 
    # `row` function (defined below).
    
    # not sure what is t and csv()
    def read(sfile):
        data = DATA()
        # not sure csv() is correct or not
        csv(sfile, function(t), row(data, t))
        return data

    # Create a new DATA with the same columns as  `data`. Optionally, load up the new
    # DATA with the rows inside `ts`.
    def clone(data, ts=None):
        data1 = row(DATA(), data["cols"]["names"])
        if ts:
            for t in ts:
                row(data1, t)
        return data1

# Update `data` with  row `t`. If `data.cols`
# does not exist, the use `t` to create `data.cols`.
# Otherwise, add `t` to `data.rows` and update the summaries in `data.cols`.
# To avoid updating skipped columns, we only iterate
# over `cols.x` and `cols.y`.
def row(data, t):
    if data["cols"]:
        #data["rows"].update(t)
        push(data['rows'], t)
        for cols in data["cols"]["x"], data["cols"]["y"]:
            for col in cols:
                #not sure what is add()
                add(col, t[col["at"]])
    else:
        data.cols = COLS(t)
    return data

# Update one COL with `x` (values from one cells of one row).
# Used  by (e.g.) the `row` and `adds` function.
# `SYM`s just increment a symbol counts.
# `NUM`s store `x` in a finite sized cache. When it
# fills to more than `the.Max`, then at probability 
# `the.Max/col.n` replace any existing item
# (selected at random). If anything is added, the list
# may not longer be sorted so set `col.ok=false`.

def add(col, x):
    if x != "?":
        n = n or 1
        col['n'] += n
        if col['isSym']:
            col['has'][x] = n + (col['has'].get(x) or 0 )
            if col['has'][x] > col['most']:
                col['most'], col['mode'] = col['has'][x], x
        else:
            col['lo'], col['hi'] = min(x, col['lo']), max(x, col['hi'])
            
            # all_ is all in lua
            # all_ and pos are local
            all_, pos = len(col['has']), None
            if all_ < the.Max:
                pos = all_ + 1
            # rand() is defined in later code
            elif rand() < the.Max / col['n']:
                pos = rint(1, all_)
            if pos is not None:
                col['has'][pos] = x
                col['ok'] = False

# Update a COL with multiple items from `t`. This is useful when `col` is being
# used outside of some DATA.
def adds(col, t):
    for x in t or {}:
        add(col, x)
    return col

# Update a RANGE to cover `x` and `y`
def extend(range_, n, s):
    range_.lo = min(n, range_.lo)
    range_.hi = max(n, range_.hi)
    add(range_.y, s)

# A query that returns contents of a column. If `col` is a `NUM` with
# unsorted contents, then sort before return the contents.
# Called by (e.g.) the `mid` and `div` functions.
def has(col):
    if not col.isSym and not col.ok:
        col.has.sort()
    col.ok = True
    return col.has

# A query that returns a `cols`'s central tendency
# (mode for `SYM`s and median for `NUM`s). Called by (e.g.) the `stats` function.
def mid(col):
    if col['isSym'] and col['mode']:
        return col['mode']
    else:
        # has_ = has(col)
        # return per(has_, 0.5)
        return per(has(col), 0.5)

# A query that returns a `col`'s deviation from central tendency
# (entropy for `SYM`s and standard deviation for `NUM`s)..
def div(col):
    if col['isSym']:
        e = 0
        for n in col['has']:
            e -= n / col['n'] * math.log(n / col['n'], 2)
        return e
    else:
        has_ = has(col)
        return (per(has_, 0.9) - per(has_, 0.1)) / 2.58

# A query that returns `mid` or `div` of `cols` (defaults to `data.cols.y`).
def stats(data, fun=None, cols=None, nPlaces=None):
    cols = cols or data['cols']['y']
    # not sure follow code is correct or not
    # rnd() is defined in later code
    tmp = {k: (rnd((fun or mid)(col), nPlaces), col['txt']) for k, col in cols.items()}
    tmp['N'] = len(data['rows'])
    return tmp, {k: mid(col) for k, col in cols.items()}

# A query that normalizes `n` 0..1. Called by (e.g.) the `dist` function.
def norm(num,n):
    # what is 'x' here for???
    
    return (n - num["lo"])/(num["hi"]-num["lo"] + 1/math.inf)
