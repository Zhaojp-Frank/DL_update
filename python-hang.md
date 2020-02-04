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
- sudo kill -SIGUSR1 <PID>
