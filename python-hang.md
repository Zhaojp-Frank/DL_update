# Python process hang debugging
- dump python process stack when it's running via pdbx, kill-signal

## download pdbx
```
wget https://files.pythonhosted.org/packages/f1/06/62aa034c16d2225424318daa20597f0d20a23ead4d147f32ed477de03544/pdbx-0.3.0.tar.gz
tar -xvf pdbx-0.3.0.tar.gz
cd pdbx-0.3.0
python setup.py install
```

## Prepare your python script
```
import pdbx
import pdb,signal,time
import threading
import sys
import os
import traceback

def debug(sig, frame):
    pdb.set_trace()
    
if __name__ == '__main__':
    # zhaojp
    pdbx.enable_pystack()
    #signal.signal(signal.SIGUSR1, debug)
```
## Run your python 

## Signal to get process stack
- sudo kill -SIGUSR1 PID
```
    ThreadPoolExecutor-0_7 tid=140147785979648
    at self.__bootstrap_inner()(threading.py:774)
    at self.run()(threading.py:801)
    at self.__target(*self.__args, **self.__kwargs)(threading.py:754)
    at work_item = work_queue.get(block=True)(thread.py:73)
    at self.not_empty.wait()(Queue.py:168)
    at waiter.acquire()(threading.py:340)

ThreadPoolExecutor-0_6 tid=140147794372352
    at self.__bootstrap_inner()(threading.py:774)
    at self.run()(threading.py:801)
    at self.__target(*self.__args, **self.__kwargs)(threading.py:754)
    at work_item = work_queue.get(block=True)(thread.py:73)
    at self.not_empty.wait()(Queue.py:168)
    at waiter.acquire()(threading.py:340)
```
