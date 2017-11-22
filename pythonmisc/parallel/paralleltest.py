from __future__ import print_function

# global imports
from multiprocessing import Pool
import sys, os
from time import sleep
import numpy as np


# local imports



def waitrounds(args):
    for i in range(args[0]):
        rand = args[1][i]
        print('round %s of %s: process %s is waiting %s' %(i,args[0],os.getpid(),rand))
        sleep(rand)
        args[2] += rand
    print('process %s is done' %os.getpid())
    print('result appended = %s' % args[2])

    return args[2]

def parallel_waitrounds(noprocesses = 4,todolist = [3,4,5,6,1]):
    print('using %s processes for %s number of tasks with these number of rounds:' % (noprocesses, len(todolist)))
    print(todolist)

    result = np.zeros(shape = (len(todolist)))
    
    for item_no, item in enumerate(todolist):
        todolist[item_no] = [item, np.random.randint(10, size = item), result[item_no]]

    print(todolist)
    pool = Pool(processes=noprocesses)
    otherresult = np.asarray(pool.map(waitrounds,todolist))
    print('done')
    print('result = ')
    print(result)
    print('otherresult = ')
    print(otherresult)

if __name__=='__main__':
    parallel_waitrounds()
