# -*- coding: utf-8 -*-

from multiprocessing import Pool
import time

#
#    q = Queue()
#    p = Process(target=f, args=(q,))
#    p.start()
#    print(q.get())    # prints "[42, None, 'hello']"
#    p.join()
    
def multimap(func, args, processes=None, sleeptime=.1):
    """
    Pool map with print output
    """
    p = Pool(processes)
    result = p.map_async(func, args)
    numleft = 1
    while numleft  > 0:
        
        new = result._number_left
        if new != numleft:
            print('Number left=%s, ' % new, end='')
        else:
            time.sleep(sleeptime)
        numleft = new
        
    p.close()
        
    return result.get()
    

    
if __name__ == '__main__':
    args = list(range(100))
    print(args)
    a = multimap(test, args , processes=8)