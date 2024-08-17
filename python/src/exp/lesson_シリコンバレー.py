#%%
print('hello')
# %%
a = 'test'
b = a
c = b
print(c)
# %%
num = 1
name = '1'
print(num,type(num))
print(name,type(name))
# %%
print('Hi', 'Mike',sep=',',end='.\n')
# %%
print(2 + 2)
# %%
2 + 2
# %%
5--2
# %%
59 - 5 * 6
# %%
(50 - 5) * 6
# %%
8 / 5
# %%
17 // 3
# %%
17 % 3
# %%
5 ** 5
# %%
x = 5
y = 10
x * y

# %%
round(17/3, 2)
# %%
import math
math.sqrt(25)
# %%
y = math.log2(10)
print(y)
# %%
print(help(math))
# %%
s = 'test'
print(s)
# %%
print('I don\'t know')
# %%
print("say \"I don't know\"")
# %%
print('hello. \nHow are you?')
# %%
print(r'C:\name\name')
# %%
print("#################")
print("""\
line1
line2
line3\
""")
print("#################")

# %%
print('Hi.' *  3 + 'Mike')
# %%
print('Py' + 'thon')
# %%
print('Py','thon')

# %%
prefix = 'Py'
print(prefix+'thon')
# %%
s = ('aaaaaaaaaaaaaaaaaaa'
     'bbbbbbbbbbbbbbbbbbbbb')
print(s)
# %%
word = 'Python'
print(word[0])
print(word[-1])
print('###############')
print(word[0:2]) #0以上2未満
print(word[:2]) 
print('###############')
print(word[2:]) 
print('###############')
word = 'j' + word[1:]
print(word[:])
n = len(word)
print(n)
# %%
s = 'My name is Mike. Hi Mike.'
print(s)
is_start = s.startswith('My')
print(is_start)
is_start = s.startswith('X')

print(is_start)
print("##############")
print(s.find('Mike'))
print(s.rfind('Mike'))
print(s.count('Mike'))
print(s.capitalize())
print(s.title())
print(s.upper())
print(s.lower())
print(s.replace('Mike', 'Nancy'))

# %%
name,family,c = 'Jun','sakai',3
print(f'My name is {name} {family}. Watashi ha {family} {name}')
# %%
a = str(1)
type(a)
# %%
l = [1,20, 4, 50, 2, 1, 2]
l[0]
l[-1]
l[2:5]
l[:]
len(l)
type(l)
# %%
list('abcde')
# %%
n = [1,2,3,4,5,6,7,8,9,10]
n[::2]
n[::-1]
# %%
a = ['a', 'b', 'c']
n = [1,2,3]
x = [a,n]
x
# %%
x[0]
x[1]
# %%
x[0][1]
# %%
x[1][2]
# %%
s = ['a','b','c','d','e','f','g']
s
# %%
s[0] = 'X'
s
# %%
s[2:5] = ['C','D','E']
s
# %%
s[2:5] = []
s
# %%
s[:] = []
s
# %%
n = [1,2,3,4,5,6,7,8,9,10]
n.append(100)
n
# %%
n.insert(0,200)
n
# %%
n.pop()
# %%
n
# %%
n.pop(0)
# %%
n
# %%
del n[0]
n
# %%
del n
# %%
n = [1,2,2,2,3]
# %%
n.remove(2)
# %%
n.remove(2)
n.remove(2)

# %%
n
# %%
n.remove(2)

# %%
a = [1,2,3,4,5]
b = [6,7,8,9,10]
x = a + b
x
# %%
a += b
a
# %%
x = [1,2,3,4,5]
y = [6,7,8,9,10]
x.extend(y)
x
# %%
r = [1,2,3,4,5,1,2,3]
r.index(3,3)
# %%
r.count(3)
# %%
if 5 in r:
  print('exist')
# %%
r.sort()
print(r)
# %%
r.sort(reverse=True)
r
# %%
r.reverse()
r
# %%
s = 'My name is Mike.'
to_split = s.split(' ')
print(to_split)
# %%
x = ' ##### '.join(to_split)
x
# %%
help(list)
# %%
i = [1,2,3,4,5]
j = i
j[0] =100
print('j=',j)
print('i=',i)
# %%
x = [1,2,3,4,5]
y = x.copy()
# y = x[:]
y[0] = 100
print('y =', y)
print('x =', x)
# %%
X = 20
Y = X
Y = 5
print(id(X))
print(id(Y))
print(Y)
print(X)
# %%
X = ['a', 'b']
Y = X
Y[0] = 'P'
print(id(X))
print(id(Y))
print(Y)
print(X)
# %%
seat = []
min = 0
max = 5
min <= len(seat) < max
# %%
seat.append('p')
min <= len(seat) < max

# %%
len(seat)
# %%
seat.append('p')
seat.append('p')
seat.append('p')

# %%
min <= len(seat) < max

# %%
seat.append('p')
min <= len(seat) < max

# %%
seat.pop(0)
# %%
min <= len(seat) < max

# %%
t = (1,2,3,4,1,2)
t
# %%
type(t)
# %%
t[0]
# %%
help(tuple)
# %%
t = ([1,2,3],[4,5,6])
# %%
t[0][0] = 100
t
# %%
t = 1,2,3
t
# %%
t = 1,
t
# %%
t = ()
t
# %%
t = (1)
t
# %%
t = ('test',)
t
# %%
t = 1,
t + 100
# %%
num_tuple = (10,20)
print(num_tuple)
# %%
x, y = num_tuple
print(x, y)
# %%
min , max = 0, 100
print(min,max)
# %%
i = 10
j = 20
i, j = j, i
print(i,j)
# %%
chose_from_two = ('A','B','C')
answer = []
answer.append('A')
answer.append('C')
print(chose_from_two)
print(answer)
# %%
