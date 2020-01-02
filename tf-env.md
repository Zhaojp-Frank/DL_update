# TF relevant ev
## lib/include flag
```
>>> print(tf.__version__)
1.12.0
>>> tf.sysconfig.get_include()
'/home/admin/workspace/anaconda2/lib/python2.7/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/home/admin/workspace/anaconda2/lib/python2.7/site-packages/tensorflow'
>>> tf.sysconfig.get_compile_flags()
['-I/home/admin/workspace/anaconda2/lib/python2.7/site-packages/tensorflow/include', '-D_GLIBCXX_USE_CXX11_ABI=0']
>>> tf.sysconfig.get_link_flags()
['-L/home/admin/workspace/anaconda2/lib/python2.7/site-packages/tensorflow', '-ltensorflow_framework']

>>> import tensorflow as tf
2020-01-02 04:01:20.005283: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.1
>>> print(tf.__version__)
1.14.0
>>> tf.sysconfig.get_include()
'/usr/local/lib/python3.6/dist-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python3.6/dist-packages/tensorflow'
>>> tf.sysconfig.get_compile_flags()
['-I/usr/local/lib/python3.6/dist-packages/tensorflow/include', '-D_GLIBCXX_USE_CXX11_ABI=0']
>>> tf.sysconfig.get_link_flags()
['-L/usr/local/lib/python3.6/dist-packages/tensorflow', '-l:libtensorflow_framework.so.1']
```

