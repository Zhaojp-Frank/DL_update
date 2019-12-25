# TF-xla
## how to enable
- TF_XLA_FLAGS=--tf_xla_auto_jit=2 python ... 
- or, tf.config.optimizer.set_jit(True)
- to enable on CPU: TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" python ..
## how to disable XLA?
- ?

# eager
## TF2.0 with eager ON by default
- to disable eager in TF2.0/TF1.:
```
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
a = tf.constant(100)
b = tf.constant(200)
c = a + b
print(c)
### shows tf.Tensor(300, shape=(), dtype=int32) if eager ON, else Tensor("add:0", shape=(), dtype=int32)
```
- to disable eager in TF1.*
```
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
```

## dump
- XLA_FLAGS="--dump_hlo_as_text --xla_dump_to=/tmp/tf-xla" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python ... 
-- then goto /tmp/tf-xla
- dump graph: TF_DUMP_GRAPH_PREFIX=/tmp/tf-xla TF_XLA_FLAGS="--tf_xla_clustering_debug"

## example
```/workspace/nvidia-examples/cnn# TF_XLA_FLAGS="--tf_xla_auto_jit=2" python ./resnet.py --layers 18 -b 2 -i 100```

### log output
tensorflow/stream_executor/cuda/ptxas_utils.cc:

```
  Step Epoch Img/sec   Loss  LR
2019-12-10 11:08:14.603812: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2019-12-10 11:08:15.313293: I tensorflow/stream_executor/cuda/ptxas_utils.cc:202] 
2019-12-10 11:08:15.467916: I tensorflow/stream_executor/cuda/ptxas_utils.cc:202] 
2019-12-10 11:08:36.425353: I tensorflow/stream_executor/cuda/ptxas_utils.cc:202] 
2019-12-10 11:08:36.473024: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10
2019-12-10 11:08:36.743096: I tensorflow/stream_executor/cuda/ptxas_utils.cc:202] 
     1   1.0     0.1  6.828  7.120 2.00000
2019-12-10 11:08:43.669968: I tensorflow/stream_executor/cuda/ptxas_utils.cc:202] 
    10  10.0     2.5  0.000  0.291 1.65620
    20  20.0    59.1  0.000  0.286 1.31220
    30  30.0    60.1  0.000  0.280 1.00820
    40  40.0    60.1  0.000  0.275 0.74420
    50  50.0    60.0  0.000  0.272 0.52020
    60  60.0    60.0  0.000  0.270 0.33620
    70  70.0    60.2  0.000  0.268 0.19220
    80  80.0    60.2  0.000  0.267 0.08820
    90  90.0    60.4  0.000  0.267 0.02420
   100 100.0    32.5  0.000  0.267 0.00020



