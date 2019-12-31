# TensorRT update
# Steps

# Different approaches
## TF graph using TRT vs. saved model for TRT engine(nvidia)
IF ops are all supported by TensorRT, importing it into TensorRT and running TensorRT directly should be **more efficient** than running it through TF-TensorRT integration. it's **hard to write native TensorRT implementations**, and usually TensorRT doesn't support all the ops. TF-TRT approach enables you to get supported parts of the networks run in TensorRT and unsupported part to run in TensorFlow,

The trade-off is then between the **ease of use** and the **performance** gain (which may be negligible, depending on the model and the way you run it)

# Run sample

# perf
![alt text](https://im0-tub-com.yandex.net/i?id=841ba9d9b7007fbd51d6fc462d2ee241&n=13 "bench")

# Ref
- https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html 
- https://github.com/NVIDIA/TensorRT  
- https://github.com/tensorflow/tensorrt 
- https://github.com/tensorflow/models/issues/4028 
- https://developer.nvidia.com/deep-learning-performance-training-inference#deeplearningperformance_inference  
- https://developer.download.nvidia.cn/video/gputechconf/gtc/2019/presentation/s9431-tensorrt-inference-with-tensorflow.pdf 
- https://medium.com/tensorflow/high-performance-inference-with-tensorrt-integration-c4d78795fbfe 

