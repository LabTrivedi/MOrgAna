import numpy as np

def decorator(func):

    def wrapper(x):
        print('I am decorating',x)
        y = func(x)
        print('it has been decorated',y)
        return y

    return wrapper

data = np.random.randint(0,1000)

@decorator
def sum1(x):
    return x+1
'''
equivalent to
sum1 = decorator(sum1)
'''

print(sum1(3))


