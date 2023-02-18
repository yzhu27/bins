def value(has, nb, nr, sGoal):
  sGoal,nB,nR = sGoal or True, nB or 1, nR or 1
  b,r = 0,0
  for x,n in pairs(has):
    if x == sGoal:
      b = b + n 
    else:
      r = r + n
  b,r = b/(nB+1/float('inf')), r/(nR+1/float('inf'))
  return b**2/(b+r)

def dist(data, t1, t2, cols):
  def dist1(col, x, y):
    if x == '?' and y == '?':
      return 1
    if col.isSym:
      return 0 if x == y else 1
    else:
      x , y = norm(col , x) , norm(col , y)
      if x == '?':
        x = 1 if y < 0.5 else 1
      if y == '?':
        y = 1 if x < 0.5 else 1
      return abs(x - y)
  d , n = 0 , 1/float('inf')
  for _ , col in enumerate(cols or data.cols.x):
    n += 1
    d += dist1(col , t1[col.at] , t2[col.at]) ** the['p']
  return (d/n)**(1/the['p'])

def better(data , row1 , row2):
  s1 = 0
  s2 = 0
  ys = data.cols.y
  for _ , col in ys.items():
      x = norm(col , row1[col.at])
      y = norm(col , row2[col.at])
      s1 -= math.exp(col.w * (x - y) / len(ys))
      s2 -= math.exp(col.w * (y - x) / len(ys))
  return (s1 / len(ys)) < (s2 / len(ys))

def half(data , rows , cols , above):
  left , right = {} , {}
  def gap(r1 , r2):
    return dist(data , r1 , r2 , cols)
  def cos(a , b , c):
    return (a**2 + c**2 - b**2)/(2*c)
  def proj(r):
    return {'row':r , 'x':cos(gap(r , A) , gap(r , B) , c)}
  rows = rows or data.rows
  some = many(rows , the['Halves'])
  A = (the['Reuse'] and above) or any(some)
  def fun(r):
    return {'row':r , 'd':gap(r,A)}
  tmp = sort(map(some , fun) , lt('d'))
  far = tmp[(len(tmp) * the['Far'])//1]
  B = far['row']
  c = far['d']
  for n , two in enumerate(sort(list(map(rows , proj).values()) , lt('x'))):
      if n+1 <= (len(rows)+1) // 2:
          push(left , two['row'])
      else:
          push(right , two['row'])
  return left , right , A , B , c

def tree(data , rows , cols , above):
  rows = rows or data.rows
  here = {'data':DATA.clone(data , rows)}
  if len(rows) >= 2*(len(data.rows)**the['min']):
    left , right , A , B = half(data , rows , cols , above)
    here['left'] = tree(data , left , cols , A)
    here['right'] = tree(data , right , cols , B)
  return here

def showTree(tree , lvl):
  if tree:
    lvl = lvl or 0
    res = fmt('%s[%s] ' , '|.. ' * lvl , str(len(tree['data'].rows)))
    if 'left' not in tree or lvl == 0:
        print(res + o(stats(tree['data'])))
    else:
        print(res)
    if 'left' in tree:
        showTree(tree['left'] , lvl + 1)
    if 'right' in tree:
        showTree(tree['right'] , lvl + 1)

def sway(data):
  def worker(rows , worse , above):
    if len(rows) <= len(data.rows)**the['min']:
      return rows , many(worse , the['rest']*len(rows))
    else:
      l , r , A , B = half(data , rows , None , above)
      if better(data , B , A):
        l , r , A , B = r , l , B , A
      def fun(row):
        push(worse , row)
      map(r , fun)
      return worker(l , worse , A)
  best , rest = worker(data.rows , {} , None)
  return DATA.clone(data , best) , DATA.clone(data , rest)

def bins(cols , rowss):
    out = {}
    for _ , col in enumerate(cols):
        ranges = {}
        for y , rows in enumerate(rowss):
            for _ , row in enumerate(rows):
                x = row[col.at]
                if x != '?':
                    k = bin(col , x)
                    ranges[k] = ranges[k] or RANGE(col.at , col.txt , x)
                    extend(ranges[k] , x , y)
        ranges = sort(map(ranges , itself) , lt('lo'))
        out[len(out)] =  ranges if col.isSym else mergeAny(ranges)
    return out

def bin(col , x):
    if x == '?' or col.isSym:
        return x
    tmp = (col.hi - col.lo) / (the['bins'] - 1)
    return 1 if col.hi == col.lo else math.floor(x/tmp + 0.5)*tmp
  
def mergeAny(ranges0):
  def noGaps(t):
    for j in range(1 , len(t)):
      t[j].lo = t[j-1].hi
    t[0].lo = float('-inf')
    t[-1].hi = float('inf')
    return t
  ranges1 , j = {} , 0
  while j < len(ranges0):
    if j < len(ranges0)-1 and ranges0[j+1]:
      left , right = ranges0[j] , ranges0[j+1]
      y = merge2(left.y , right.y)
      if y:
        j += 1
        left.hi , left.y = right.hi , right.y
    else:
      left = ranges0[j]
    push(ranges1 , left)
    j += 1
  return noGaps(ranges0) if len(ranges0) == len(ranges1) else mergeAny(ranges1)

def merge2(col1 , col2):
  new = merge(col1 , col2):
  if div(new) <= (div(col1)*col1.n + div(col2)*col2.n)/new.n:
    return new

def merge(col1 , col2):
  new = copy(col1)
  if col1.isSym:
    for x , n in col2.has:
      add(new , x , n)
  else:
    for _ , n in col2.has:
      add(new , n)
    new.lo = min(col1.lo , col2.lo)
    new.hi = max(col1.hi , col2.hi)
  return new

def itself(x):
  return x

def rnd(n , nPlaces):
  mult = 10**(nPlaces or 2)
  return math.floor(n * mult + 0.5) / mult

Seed=937162211
def rint(nlo,nhi): 
  return math.floor(0.5 + rand(nlo,nhi))

def rand(nlo,nhi):
  nlo, nhi = nlo or 0, nhi or 1
  Seed = (16807 * Seed) % 2147483647
  return nlo + (nhi-nlo) * Seed / 2147483647


