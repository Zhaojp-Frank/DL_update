# TF build out
- no static cudart in nv build images l.14/13 
- no static cudart in nv build images l.12, but use static since 1.13.1, due to tf2xla
```
$ sudo docker images|grep 1.12
daocloud.io/daocloud/tensorflow        1.12.0-devel-gpu-py3               d6c139d2fdbf        15 months ago       3.77GB
tensorflow/tensorflow                  1.12.0-devel-gpu                   0c27be65b63d        15 months ago       3.71GB
zhaojp@zhaojp-linux:/opt/tf-src/tensorflow-1.12.0/build$ sudo nvidia-docker run -it --name tf-1.12 --net=host -v /opt/:/github tensorflow/tensorflow:1.12.0-devel-gpu bash
root@zhaojp-linux:~# nm -s /usr/local/lib/python2.7/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so |grep cudaGet
                 U cudaGetDevice@@libcudart.so.9.0
                 U cudaGetDeviceProperties@@libcudart.so.9.0
                 U cudaGetErrorString@@libcudart.so.9.0
                 U cudaGetLastError@@libcudart.so.9.0

$ sudo nvidia-docker run -it --name tf-1.12 --net=host -v /opt/:/github tensorflow/tensorflow:1.12.0-devel-gpu-py3 bash
root@zhaojp-linux:~# n - /usr/local/lib/python3.5/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so |grep cudaGet
bash: n: command not found
root@zhaojp-linux:~# nm -s /usr/local/lib/python3.5/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so |grep cudaGet
                 U cudaGetDevice@@libcudart.so.9.0
                 U cudaGetDeviceProperties@@libcudart.so.9.0
                 U cudaGetErrorString@@libcudart.so.9.0
                 U cudaGetLastError@@libcudart.so.9.0
$ sudo nvidia-docker run -it --name tf-1.13 --net=host -v /opt/:/github tensorflow/tensorflow:1.13.1-gpu-py3 bash
# nm -s /usr/local/lib/python3.5/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so |grep cudaGet
000000000486f940 t _GLOBAL__I___cudaGetExportTableInternal
0000000009749884 r _ZZ13cudaGetDeviceE12__FUNCTION__
0000000009749960 r _ZZ16cudaGetErrorNameE12__FUNCTION__
00000000097499a0 r _ZZ16cudaGetLastErrorE12__FUNCTION__
```

##  tf 1.15
```
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# find . -type f|xargs grep cudaGet
Binary file ./libtensorflow_framework.so.1.15.2 matches
Binary file ./stream_executor/cuda/libcudart_stub.pic.a matches
Binary file ./stream_executor/cuda/libcuda_driver.pic.a matches
Binary file ./stream_executor/cuda/_objs/cuda_driver/cuda_driver.pic.o matches
Binary file ./stream_executor/cuda/_objs/cudart_stub/cudart_stub.pic.o matches
Binary file ./python/_pywrap_tensorflow_internal.so matches
Binary file ./lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so matches
Binary file ./core/grappler/clusters/libutils.pic.a matches
Binary file ./core/grappler/clusters/_objs/utils/utils.pic.o matches
Binary file ./core/libgpu_runtime_impl.pic.lo matches
Binary file ./core/_objs/gpu_runtime_impl/gpu_device.pic.o matches
Binary file ./core/nccl/libnccl_lib.pic.lo matches
Binary file ./core/nccl/_objs/nccl_lib/nccl_manager.pic.o matches
Binary file ./compiler/tf2tensorrt/libtrt_conversion.pic.lo matches
Binary file ./compiler/tf2tensorrt/_objs/trt_conversion/convert_graph.pic.o matches
Binary file ./compiler/tf2xla/ops/_xla_ops.so matches

Binary file ./core/kernels/libtridiagonal_matmul_op_gpu.pic.lo matches
Binary file ./core/kernels/libpopulation_count_op_gpu.pic.lo matches
Binary file ./core/kernels/libscatter_op_gpu.pic.lo matches
Binary file ./core/kernels/libresize_nearest_neighbor_op_gpu.pic.lo matches
Binary file ./core/kernels/libmatrix_set_diag_op_gpu.pic.lo matches
Binary file ./core/kernels/libscatter_nd_op_gpu.pic.lo matches
Binary file ./core/kernels/libgather_nd_op_gpu.pic.lo matches
Binary file ./core/kernels/libhistogram_op_gpu.pic.lo matches
Binary file ./core/kernels/libfused_batch_norm_op_gpu.pic.lo matches
Binary file ./core/kernels/rnn/liblstm_ops_gpu.pic.lo matches
Binary file ./core/kernels/rnn/_objs/lstm_ops_gpu/lstm_ops_gpu.cu.pic.o matches
Binary file ./core/kernels/libreduction_ops_gpu.pic.lo matches
Binary file ./core/kernels/librelu_op_gpu.pic.lo matches
Binary file ./core/kernels/libdeterminant_op_gpu.pic.lo matches
Binary file ./core/kernels/libadjust_saturation_op_gpu.pic.lo matches
Binary file ./core/kernels/libsvd_op_gpu.pic.lo matches
Binary file ./core/kernels/libscatter_functor_gpu.pic.lo matches
Binary file ./core/kernels/libsparse_tensor_dense_matmul_op_gpu.pic.lo matches
Binary file ./core/kernels/libsparse_xent_op_gpu.pic.lo matches
Binary file ./core/kernels/libdiag_op_gpu.pic.lo matches
Binary file ./core/kernels/libtopk_op_gpu.pic.lo matches
Binary file ./core/kernels/libresize_bilinear_op_gpu.pic.lo matches
Binary file ./core/kernels/libconcat_lib_gpu.pic.lo matches
Binary file ./core/kernels/libdepth_space_ops_gpu.pic.lo matches
Binary file ./core/kernels/libwhere_op_gpu.pic.lo matches
Binary file ./core/kernels/libsegment_reduction_ops_gpu.pic.lo matches
Binary file ./core/kernels/libtridiagonal_solve_op_gpu.pic.lo matches
Binary file ./core/kernels/liblu_op_gpu.pic.lo matches
Binary file ./core/kernels/libbatch_space_ops_gpu.pic.lo matches
Binary file ./core/kernels/libcrop_and_resize_op_gpu.pic.lo matches
Binary file ./core/kernels/libtranspose_functor_gpu.pic.lo matches
Binary file ./core/kernels/libbucketize_op_gpu.pic.lo matches
Binary file ./core/kernels/libroll_op_gpu.pic.lo matches
Binary file ./core/kernels/libinplace_ops_gpu.pic.lo matches
Binary file ./core/kernels/libparameterized_truncated_normal_op_gpu.pic.lo matches
Binary file ./core/kernels/libcwise_op_gpu.pic.lo matches
Binary file ./core/kernels/libmatrix_band_part_op_gpu.pic.lo matches
Binary file ./core/kernels/libsoftmax_op_gpu.pic.lo matches
Binary file ./core/kernels/libscan_ops_gpu.pic.lo matches
Binary file ./core/kernels/libbincount_op_gpu.pic.lo matches
Binary file ./core/kernels/libin_topk_op_gpu.pic.lo matches
Binary file ./core/kernels/libmultinomial_op_gpu.pic.lo matches
Binary file ./core/kernels/librandom_op_gpu.pic.lo matches
Binary file ./core/kernels/libmatrix_diag_op_gpu.pic.lo matches
Binary file ./core/kernels/libdilation_ops_gpu.pic.lo matches
Binary file ./core/kernels/libdynamic_partition_op_gpu.pic.lo matches
Binary file ./core/kernels/libpooling_ops_gpu.pic.lo matches
Binary file ./core/kernels/libconv_2d_gpu.pic.lo matches
Binary file ./core/kernels/libsearchsorted_op_gpu.pic.lo matches
Binary file ./core/kernels/libsplit_lib_gpu.pic.lo matches
Binary file ./core/kernels/_objs/concat_lib_gpu/concat_lib_gpu_impl.cu.pic.o matches
Binary file ./core/kernels/_objs/stateful_random_ops_gpu/stateful_random_ops_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/bias_op_gpu/bias_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/dilation_ops_gpu/dilation_ops_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/cwise_op_gpu/cwise_op_clip_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/pooling_ops_gpu/pooling_ops_3d_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/pooling_ops_gpu/avgpooling_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/pooling_ops_gpu/maxpooling_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/depthwise_conv_op_gpu/depthwise_conv_op_gpu_double.cu.pic.o matches
Binary file ./core/kernels/_objs/depthwise_conv_op_gpu/depthwise_conv_op_gpu_float.cu.pic.o matches
Binary file ./core/kernels/_objs/depthwise_conv_op_gpu/depthwise_conv_op_gpu_half.cu.pic.o matches
Binary file ./core/kernels/_objs/adjust_hue_op_gpu/adjust_hue_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/segment_reduction_ops_gpu/segment_reduction_ops_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/eye_functor_gpu/eye_functor_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/check_numerics_op_gpu/check_numerics_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/scatter_nd_op_gpu/scatter_nd_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/resize_nearest_neighbor_op_gpu/resize_nearest_neighbor_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/depth_space_ops_gpu/depthtospace_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/depth_space_ops_gpu/spacetodepth_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/tridiagonal_matmul_op_gpu/tridiagonal_matmul_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/inplace_ops_gpu/inplace_ops_functor_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/parameterized_truncated_normal_op_gpu/parameterized_truncated_normal_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/fused_batch_norm_op_gpu/fused_batch_norm_op.cu.pic.o matches
Binary file ./core/kernels/_objs/histogram_op_gpu/histogram_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/bincount_op_gpu/bincount_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/matrix_diag_op_gpu/matrix_diag_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_8.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_1.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_3.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_2.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_6.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_5.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_7.cu.pic.o matches
Binary file ./core/kernels/_objs/where_op_gpu/where_op_gpu_impl_4.cu.pic.o matches
Binary file ./core/kernels/_objs/matrix_set_diag_op_gpu/matrix_set_diag_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/lu_op_gpu/lu_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/gather_nd_op_gpu/gather_nd_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/in_topk_op_gpu/in_topk_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/sparse_tensor_dense_matmul_op_gpu/sparse_tensor_dense_matmul_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/roll_op_gpu/roll_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/l2loss_op_gpu/l2loss_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/scatter_op_gpu/scatter_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/crop_and_resize_op_gpu/crop_and_resize_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/split_lib_gpu/split_lib_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_int.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_uint16.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_half.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_double.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_uint64.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_uint8.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_float.cu.pic.o matches
Binary file ./core/kernels/_objs/conv_2d_gpu/conv_2d_gpu_uint32.cu.pic.o matches
Binary file ./core/kernels/_objs/svd_op_gpu/svd_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/scatter_functor_gpu/scatter_functor_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/softmax_op_gpu/softmax_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/dynamic_partition_op_gpu/dynamic_partition_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/searchsorted_op_gpu/searchsorted_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/adjust_saturation_op_gpu/adjust_saturation_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/compare_and_bitpack_op_gpu/compare_and_bitpack_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/non_max_suppression_op_gpu/non_max_suppression_op.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_int32.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_float.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_bool.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_complex64.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_half.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_double.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_int16.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_int64.cu.pic.o matches
Binary file ./core/kernels/_objs/tile_ops_gpu/tile_functor_gpu_complex128.cu.pic.o matches
Binary file ./core/kernels/_objs/multinomial_op_gpu/multinomial_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_gpu_complex128.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_gpu_float.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_half_prod_max_min.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_gpu_double.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_half_mean_sum.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_gpu_complex64.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_gpu_int.cu.pic.o matches
Binary file ./core/kernels/_objs/reduction_ops_gpu/reduction_ops_gpu_bool.cu.pic.o matches
Binary file ./core/kernels/_objs/gather_functor_gpu/gather_functor_batched_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/gather_functor_gpu/gather_functor_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/sparse_xent_op_gpu/sparse_xent_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/population_count_op_gpu/population_count_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_uint8.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_double.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_float.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_int32.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_int64.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_half.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_int16.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_uint16.cu.pic.o matches
Binary file ./core/kernels/_objs/topk_op_gpu/topk_op_gpu_int8.cu.pic.o matches
Binary file ./core/kernels/_objs/scan_ops_gpu/scan_ops_gpu_int.cu.pic.o matches
Binary file ./core/kernels/_objs/scan_ops_gpu/scan_ops_gpu_double.cu.pic.o matches
Binary file ./core/kernels/_objs/scan_ops_gpu/scan_ops_gpu_float.cu.pic.o matches
Binary file ./core/kernels/_objs/scan_ops_gpu/scan_ops_gpu_half.cu.pic.o matches
Binary file ./core/kernels/_objs/resize_bilinear_op_gpu/resize_bilinear_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/matrix_band_part_op_gpu/matrix_band_part_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/random_op_gpu/random_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/determinant_op_gpu/determinant_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/bucketize_op_gpu/bucketize_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/batch_space_ops_gpu/spacetobatch_functor_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/diag_op_gpu/diag_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/tridiagonal_solve_op_gpu/tridiagonal_solve_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/relu_op_gpu/relu_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/dynamic_stitch_op_gpu/dynamic_stitch_op_gpu.cu.pic.o matches
Binary file ./core/kernels/_objs/transpose_functor_gpu/transpose_functor_gpu.cu.pic.o matches
Binary file ./core/kernels/libdynamic_stitch_op_gpu.pic.lo matches
Binary file ./core/kernels/libcheck_numerics_op_gpu.pic.lo matches
Binary file ./core/kernels/libnon_max_suppression_op_gpu.pic.lo matches
Binary file ./core/kernels/libbias_op_gpu.pic.lo matches
Binary file ./core/kernels/libtile_ops_gpu.pic.lo matches
Binary file ./core/kernels/libadjust_hue_op_gpu.pic.lo matches
Binary file ./core/kernels/libeye_functor_gpu.pic.lo matches
Binary file ./core/kernels/libgather_functor_gpu.pic.lo matches
Binary file ./core/kernels/libl2loss_op_gpu.pic.lo matches
Binary file ./core/kernels/libdepthwise_conv_op_gpu.pic.lo matches
Binary file ./core/kernels/libcompare_and_bitpack_op_gpu.pic.lo matches
Binary file ./core/kernels/libstateful_random_ops_gpu.pic.lo matches

Binary file ./contrib/hadoop/_dataset_ops.so matches
Binary file ./contrib/tensor_forest/python/ops/_model_ops.so matches
Binary file ./contrib/tensor_forest/python/ops/_stats_ops.so matches
Binary file ./contrib/tensor_forest/python/ops/_tensor_forest_ops.so matches
Binary file ./contrib/kinesis/_dataset_ops.so matches
Binary file ./contrib/reduce_slice_ops/python/ops/_reduce_slice_ops.so matches
Binary file ./contrib/reduce_slice_ops/python/ops/python/ops/lib_reduce_slice_ops_gpu.pic.a matches
Binary file ./contrib/reduce_slice_ops/_objs/python/ops/_reduce_slice_ops_gpu/reduce_slice_ops_gpu.cu.pic.o matches
Binary file ./contrib/kafka/_dataset_ops.so matches
Binary file ./contrib/boosted_trees/python/ops/_boosted_trees_ops.so matches
Binary file ./contrib/libsvm/python/ops/_libsvm_ops.so matches
Binary file ./contrib/bigtable/python/ops/_bigtable.so matches
Binary file ./contrib/resampler/python/ops/python/ops/lib_resampler_ops_gpu.pic.a matches
Binary file ./contrib/resampler/python/ops/_resampler_ops.so matches
Binary file ./contrib/resampler/_objs/python/ops/_resampler_ops_gpu/resampler_ops_gpu.cu.pic.o matches
Binary file ./contrib/image/python/ops/_image_ops.so matches

Binary file ./contrib/ignite/_ignite_ops.so matches
Binary file ./contrib/framework/python/ops/_variable_ops.so matches
Binary file ./contrib/layers/python/ops/_sparse_feature_cross_op.so matches
Binary file ./contrib/text/python/ops/_skip_gram_ops.so matches
Binary file ./contrib/factorization/python/ops/_factorization_ops.so matches
Binary file ./contrib/fused_conv/python/ops/_fused_conv2d_bias_activation_op.so matches
Binary file ./contrib/nearest_neighbor/python/ops/_nearest_neighbor_ops.so matches
Binary file ./contrib/ffmpeg/ffmpeg.so matches

```

## some cases
from tf2xla/ops/_xla_ops.so
```
#tf.1.15.2 build myself
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# find . -type f -name "*.lo"|xargs grep _ZZ13cudaGetDeviceE12__FUNCTION_
_root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# find . -type f -name "*.o"|xargs grep _ZZ13cudaGetDeviceE12__FUNCTION__


# find . -type f -name "*.so*"|xargs grep _ZZ13cudaGetDeviceE12__FUNCTION__
Binary file ./python/_pywrap_tensorflow_internal.so matches
Binary file ./lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so matches
Binary file ./compiler/tf2xla/ops/_xla_ops.so matches <<====

Binary file ./contrib/hadoop/_dataset_ops.so matches
Binary file ./contrib/tensor_forest/python/ops/_model_ops.so matches
Binary file ./contrib/tensor_forest/python/ops/_stats_ops.so matches
Binary file ./contrib/tensor_forest/python/ops/_tensor_forest_ops.so matches
Binary file ./contrib/kinesis/_dataset_ops.so matches
Binary file ./contrib/reduce_slice_ops/python/ops/_reduce_slice_ops.so matches
Binary file ./contrib/kafka/_dataset_ops.so matches
Binary file ./contrib/boosted_trees/python/ops/_boosted_trees_ops.so matches
Binary file ./contrib/libsvm/python/ops/_libsvm_ops.so matches
Binary file ./contrib/bigtable/python/ops/_bigtable.so matches
Binary file ./contrib/resampler/python/ops/_resampler_ops.so matches
Binary file ./contrib/image/python/ops/_image_ops.so matches
Binary file ./contrib/image/python/ops/_single_image_random_dot_stereograms.so matches
Binary file ./contrib/image/python/ops/_distort_image_ops.so matches
Binary file ./contrib/memory_stats/python/ops/_memory_stats_ops.so matches
Binary file ./contrib/seq2seq/python/ops/_beam_search_ops.so matches
Binary file ./contrib/input_pipeline/python/ops/_input_pipeline_ops.so matches
Binary file ./contrib/periodic_resample/python/ops/_periodic_resample_op.so matches
Binary file ./contrib/ignite/_ignite_ops.so matches
Binary file ./contrib/framework/python/ops/_variable_ops.so matches
Binary file ./contrib/layers/python/ops/_sparse_feature_cross_op.so matches
Binary file ./contrib/text/python/ops/_skip_gram_ops.so matches
Binary file ./contrib/factorization/python/ops/_factorization_ops.so matches
Binary file ./contrib/fused_conv/python/ops/_fused_conv2d_bias_activation_op.so matches
Binary file ./contrib/nearest_neighbor/python/ops/_nearest_neighbor_ops.so matches
Binary file ./contrib/ffmpeg/ffmpeg.so matches

/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# nm -s ./compiler/tf2xla/ops/_xla_ops.so|grep cudaGet
00000000001ca5b0 t _GLOBAL__I___cudaGetExportTableInternal
0000000000244784 r _ZZ13cudaGetDeviceE12__FUNCTION__
0000000000244860 r _ZZ16cudaGetErrorNameE12__FUNCTION__
00000000002448a0 r _ZZ16cudaGetLastErrorE12__FUNCTION__
0000000000243c10 r _ZZ17cudaGetSymbolSizeE12__FUNCTION__
00000000002437f0 r _ZZ18cudaGetChannelDescE12__FUNCTION__
0000000000244820 r _ZZ18cudaGetDeviceCountE12__FUNCTION__
0000000000244730 r _ZZ18cudaGetDeviceFlagsE12__FUNCTION__
0000000000244840 r _ZZ18cudaGetErrorStringE12__FUNCTION__
0000000000243c30 r _ZZ20cudaGetSymbolAddressE12__FUNCTION__
0000000000244800 r _ZZ23cudaGetDevicePropertiesE12__FUNCTION__
0000000000243810 r _ZZ23cudaGetSurfaceReferenceE12__FUNCTION__
0000000000243850 r _ZZ23cudaGetTextureReferenceE12__FUNCTION__
0000000000243f60 r _ZZ26cudaGetMipmappedArrayLevelE12__FUNCTION__
0000000000243870 r _ZZ29cudaGetTextureAlignmentOffsetE12__FUNCTION__
0000000000243740 r _ZZ31cudaGetTextureObjectTextureDescE12__FUNCTION__
0000000000243680 r _ZZ32cudaGetSurfaceObjectResourceDescE12__FUNCTION__
0000000000243760 r _ZZ32cudaGetTextureObjectResourceDescE12__FUNCTION__
0000000000243700 r _ZZ36cudaGetTextureObjectResourceViewDescE12__FUNCTION__
00000000001ca5c0 t __cudaGetExportTableInternal
00000000001d2010 t cudaGetChannelDesc
00000000001dcc70 t cudaGetDevice
00000000001dd700 t cudaGetDeviceCount
00000000001dc780 t cudaGetDeviceFlags


./python/_pywrap_tensorflow_internal.so
nm -s ./python/_pywrap_tensorflow_internal.so|grep cudaGet
0000000006ce02e0 t _GLOBAL__I___cudaGetExportTableInternal
0000000008a49b64 r _ZZ13cudaGetDeviceE12__FUNCTION__
0000000008a49c40 r _ZZ16cudaGetErrorNameE12__FUNCTION__
0000000008a49c80 r _ZZ16cudaGetLastErrorE12__FUNCTION__
0000000008a48ff0 r _ZZ17cudaGetSymbolSizeE12__FUNCTION__
0000000008a48bd0 r _ZZ18cudaGetChannelDescE12__FUNCTION__
0000000008a49c00 r _ZZ18cudaGetDeviceCountE12__FUNCTION__
0000000008a49b10 r _ZZ18cudaGetDeviceFlagsE12__FUNCTION__
0000000008a49c20 r _ZZ18cudaGetErrorStringE12__FUNCTION__
0000000008a49010 r _ZZ20cudaGetSymbolAddressE12__FUNCTION__
0000000008a49be0 r _ZZ23cudaGetDevicePropertiesE12__FUNCTION__
0000000008a48bf0 r _ZZ23cudaGetSurfaceReferenceE12__FUNCTION__
0000000008a48c30 r _ZZ23cudaGetTextureReferenceE12__FUNCTION__
0000000008a49340 r _ZZ26cudaGetMipmappedArrayLevelE12__FUNCTION__
0000000008a48c50 r _ZZ29cudaGetTextureAlignmentOffsetE12__FUNCTION__
0000000008a48b20 r _ZZ31cudaGetTextureObjectTextureDescE12__FUNCTION__
0000000008a48a60 r _ZZ32cudaGetSurfaceObjectResourceDescE12__FUNCTION__
0000000008a48b40 r _ZZ32cudaGetTextureObjectResourceDescE12__FUNCTION__
0000000008a48ae0 r _ZZ36cudaGetTextureObjectResourceViewDescE12__FUNCTION__
0000000006ce02f0 t __cudaGetExportTableInternal
0000000006ce7d40 t cudaGetChannelDesc
0000000006cf29a0 t cudaGetDevice

```

## how to
tf2xla add "libcudart_static" but actualy useless ?
```
# grep "libcudart_static" compiler/tf2xla/ops/_xla_ops.so-2.params 
bazel-out/k8-opt/bin/external/local_config_cuda/cuda/cuda/lib/libcudart_static.a

# grep 'cuda' *
grep: _objs: Is a directory
Binary file _xla_ops.so matches
_xla_ops.so-2.params:bazel-out/k8-opt/bin/external/local_config_cuda/cuda/cuda/lib/libcudart_static.a

/opt/tf-src/tensorflow-1.15.2/tensorflow/compiler/tf2xla$ find . -type f |xargs grep -Hi cuda
./BUILD:load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_cuda_cc_test")
./BUILD:    "//tensorflow/core/platform:default/cuda_build_defs.bzl",
./BUILD:    "if_cuda_is_configured",
./BUILD:    ] + if_cuda_is_configured([
./BUILD:        "//tensorflow/core:stream_executor_no_cuda",
./BUILD:tf_cuda_cc_test(
./BUILD:tf_cuda_cc_test(
./xla_op_registry.cc:#include "tensorflow/core/platform/stream_executor_no_cuda.h"


root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# ll compiler/tf2xla/ops/
total 3316
drwxr-xr-x 3 root root    4096 Jan 30 16:16 ./
drwxr-xr-x 3 root root    4096 Jan 30 15:02 ../
drwxr-xr-x 3 root root    4096 Jan 30 15:02 _objs/
-r-xr-xr-x 1 root root 3259600 Jan 30 16:16 _xla_ops.so*
-r-xr-xr-x 1 root root    4372 Jan 30 15:02 _xla_ops.so-2.params*
-r-xr-xr-x 1 root root  110860 Jan 30 16:10 gen_xla_ops.py*
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# nm -s compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.|grep cudaGet
xla_ops.pic.d  xla_ops.pic.o  
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# nm -s compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o|grep cudaGet
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# ll compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o|grep cudaGet
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# grep cudaGet compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# ll compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o
-r-xr-xr-x 1 root root 108992 Jan 30 15:02 compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o*
root@zhaojp-linux:/github/tf-src/tensorflow-1.15.2/bazel-bin/tensorflow# ll compiler/tf2xla/ops/_xla_ops.so
-r-xr-xr-x 1 root root 3259600 Jan 30 16:16 compiler/tf2xla/ops/_xla_ops.so*


