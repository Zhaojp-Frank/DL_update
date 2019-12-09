# paper
# Using TFT
Ref: https://github.com/tensorflow/tensorrt/tree/master/tftrt/examples/image-classification
nvidia-tf:19.09-py3
model: 
- mobilenet_v2: url: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
### cmd
```
export PYTHONPATH="$PYTHONPATH:/workspace/nvidia-examples/tensorrt/tftrt/examples/third_party/models/"; \
python image_classification.py \
    --use_synthetic \
    --model_dir /models/tf-model/mobilenet_v2_1.4_224/ \
    --model mobilenet_v2 \
    --mode benchmark \
    --num_warmup_iterations 30 \
    --display_every 10 \
    --use_trt \
    --precision fp16 \
    --max_workspace_size $((2**32)) \
    --batch_size 8 \
	--num_iterations 50
  ```
  ### Output
  ``` 
2019-12-09 08:57:31.047898: I tensorflow/compiler/tf2tensorrt/segment/segment.cc:460] There are 7 ops of 6 different types in the graph that are not converted to TensorRT: Add, ArgMax, Identity, Const, NoOp, Placeholder, (For more information see https://docs.nvidia.com/deeplearning/dgx/tf-trt-user-guide/index.html#supported-ops).
2019-12-09 08:57:31.076361: I tensorflow/compiler/tf2tensorrt/convert/convert_graph.cc:735] Number of TensorRT candidate segments: 1
2019-12-09 08:57:31.139157: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.1
2019-12-09 08:57:31.316127: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2019-12-09 08:57:32.417454: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10
2019-12-09 08:58:13.564371: I tensorflow/compiler/tf2tensorrt/convert/convert_graph.cc:837] TensorRT node TRTEngineOp_0 added for segment 0 consisting of 507 nodes succeeded.
2019-12-09 08:58:13.716197: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:752] Optimization results for grappler item: tf_graph
2019-12-09 08:58:13.716239: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:754]   constant folding: Graph size after: 509 nodes (-262), 518 edges (-262), time = 131.381ms.
2019-12-09 08:58:13.716247: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:754]   layout: Graph size after: 513 nodes (4), 522 edges (4), time = 50.402ms.
2019-12-09 08:58:13.716253: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:754]   constant folding: Graph size after: 513 nodes (0), 522 edges (0), time = 62.44ms.
2019-12-09 08:58:13.716258: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:754]   TensorRTOptimizer: Graph size after: 7 nodes (-506), 6 edges (-516), time = 42673.7617ms.
batch_size: 8
cache: False
calib_data_dir: None
data_dir: None
default_models_dir: ./data
display_every: 10
engine_dir: None
max_workspace_size: 4294967296
minimum_segment_size: 2
mode: benchmark
model: mobilenet_v2
model_dir: /models/tf-model/mobilenet_v2_1.4_224/
num_calib_inputs: 500
num_iterations: 50
num_warmup_iterations: 30
precision: FP16
target_duration: None
use_synthetic: True
use_trt: True
use_trt_dynamic_op: False
url: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
num_nodes(native_tf): 771
num_nodes(tftrt_total): 7
num_nodes(trt_only): 1
graph_size(MB)(native_tf): 23.6
graph_size(MB)(trt): 36.0
time(s)(trt_conversion): 43.7
running inference...
WARNING:tensorflow:Input graph does not use tf.data.Dataset or contain a QueueRunner. That means predict yields forever. This is probably a mistake.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/array_ops.py:1354: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2019-12-09 08:58:14.937368: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties:
name: Tesla P100-PCIE-16GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:04:00.0
2019-12-09 08:58:14.938556: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 1 with properties:
name: Tesla P100-PCIE-16GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:84:00.0
2019-12-09 08:58:14.938639: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.1
2019-12-09 08:58:14.938792: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10
2019-12-09 08:58:14.938817: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcufft.so.10
2019-12-09 08:58:14.938850: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcurand.so.10
2019-12-09 08:58:14.938938: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusolver.so.10
2019-12-09 08:58:14.938972: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusparse.so.10
2019-12-09 08:58:14.938994: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2019-12-09 08:58:14.942919: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0, 1
2019-12-09 08:58:14.942992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-12-09 08:58:14.943004: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0 1
2019-12-09 08:58:14.943013: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N N
2019-12-09 08:58:14.943022: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 1:   N N
2019-12-09 08:58:14.945984: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15099 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:04:00.0, compute capability: 6.0)
2019-12-09 08:58:14.947255: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 15099 MB memory) -> physical GPU (device: 1, name: Tesla P100-PCIE-16GB, pci bus id: 0000:84:00.0, compute capability: 6.0)
2019-12-09 08:58:15.122693: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
    step 10/50, iter_time(ms)=4.4785, images/sec=1786
    step 20/50, iter_time(ms)=4.4987, images/sec=1778
    step 30/50, iter_time(ms)=4.4539, images/sec=1796
    step 40/50, iter_time(ms)=4.4658, images/sec=1791
    step 50/50, iter_time(ms)=4.4386, images/sec=1802
results of mobilenet_v2:
    images/sec: 1798
    99th_percentile(ms): 4.48
    total_time(s): 0.1
    latency_mean(ms): 4.45
    latency_median(ms): 4.45
    latency_min(ms): 4.41

```

## MPSissue
TF launch failed if MPS is enabled
```
2019-12-09 10:29:08.537032: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x497c5f0 executing computations on platform Host. Devices:
2019-12-09 10:29:08.537052: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-12-09 10:29:08.539277: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcuda.so.1
2019-12-09 10:29:08.549054: W tensorflow/compiler/xla/service/platform_util.cc:256] unable to create StreamExecutor for CUDA:0: failed initializing StreamExecutor for CUDA device ordinal 0: Internal: failed call to cuDevicePrimaryCtxRetain: CUDA_ERROR_INVALID_DEVICE: invalid device ordinal
2019-12-09 10:29:08.549116: F tensorflow/stream_executor/lib/statusor.cc:34] Attempting to fetch value instead of handling error Internal: no supported devices found for platform CUDA
./bench.sh: line 14:   389 Aborted                 python image_classification.py --use_synthetic --model_dir /models/tf-model/mobilenet_v2_1.4_224/ --model mobilenet_v2 --mode benchmark --num_warmup_iterations 20 --display_every 10 --precision $f --max_workspace_size $((2**32)) --batch_size $b --num_iterations 50 $use_trt
root@r67b13166:/workspace/nvidia-examples/tensorrt/tftrt/examples/image-classification# nvidia-smi
Mon Dec  9 10:30:05 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:1B:00.0 Off |                    0 |
| N/A   28C    P0    43W / 300W |     40MiB / 16130MiB |      0%   E. Process |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
