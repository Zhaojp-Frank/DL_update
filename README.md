#build tf2xla

## xla_ops
```
root@zhaojp-linux:/github/tf-src/tensorflow-1.14.0# find bazel-out/ -type f -name _pywrap_tensorflow_internal.so
bazel-out/host/bin/tensorflow/python/_pywrap_tensorflow_internal.so
root@zhaojp-linux:/github/tf-src/tensorflow-1.14.0# find bazel-out/ -type f |xargs grep "_xla_ops"
bazel-out/k8-opt/bin/tensorflow/python/debug/grpc_tensorflow_server.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/debug/grpc_tensorflow_server.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/import_pb_to_tensorboard.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/import_pb_to_tensorboard.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/print_selective_registration_header.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/print_selective_registration_header.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/saved_model_cli.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/saved_model_cli.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/inspect_checkpoint.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/inspect_checkpoint.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/import_pb_to_tensorboard.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/import_pb_to_tensorboard.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/optimize_for_inference.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/optimize_for_inference.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/freeze_graph.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/freeze_graph.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/optimize_for_inference.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/optimize_for_inference.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/freeze_graph.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/freeze_graph.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/print_selective_registration_header.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/print_selective_registration_header.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/inspect_checkpoint.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/inspect_checkpoint.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/python/tools/saved_model_cli.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/k8-opt/bin/tensorflow/python/tools/saved_model_cli.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/lite/python/tflite_convert.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/lite/python/tflite_convert.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so-2.params:bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o
Binary file bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/libxla_ops.pic.lo matches
Binary file bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_objs/xla_ops/xla_ops.pic.o matches
Binary file bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o matches
bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.d:bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o: \
bazel-out/k8-opt/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py:Original C++ source file: gen_xla_ops.cc
Binary file bazel-out/host/bin/tensorflow/python/_pywrap_tensorflow_internal.so matches
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_2_keras_python_api_gen_compat_v2.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_2_keras_python_api_gen_compat_v2.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen_compat_v1.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen_compat_v1.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen_compat_v1.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_1_keras_python_api_gen_compat_v1.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_2_keras_python_api_gen_compat_v2.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/python/keras/api/create_tensorflow.python_api_2_keras_python_api_gen_compat_v2.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
Binary file bazel-out/host/bin/tensorflow/compiler/jit/ops/libxla_ops.pic.lo matches
bazel-out/host/bin/tensorflow/compiler/jit/ops/gen_xla_ops_wrapper_py_py_wrappers_cc.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/jit/ops/gen_xla_ops_wrapper_py_py_wrappers_cc /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/jit/ops/gen_xla_ops_wrapper_py_py_wrappers_cc
Binary file bazel-out/host/bin/tensorflow/compiler/jit/ops/gen_xla_ops_wrapper_py_py_wrappers_cc matches
bazel-out/host/bin/tensorflow/compiler/jit/ops/gen_xla_ops_wrapper_py_py_wrappers_cc.runfiles_manifest:org_tensorflow/tensorflow/compiler/jit/ops/gen_xla_ops_wrapper_py_py_wrappers_cc /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/jit/ops/gen_xla_ops_wrapper_py_py_wrappers_cc
Binary file bazel-out/host/bin/tensorflow/compiler/jit/ops/libxla_ops.lo matches
Binary file bazel-out/host/bin/tensorflow/compiler/jit/ops/_objs/xla_ops/xla_ops.pic.o matches
Binary file bazel-out/host/bin/tensorflow/compiler/jit/ops/_objs/xla_ops/xla_ops.o matches
Binary file bazel-out/host/bin/tensorflow/compiler/jit/kernels/libxla_ops.pic.lo matches
Binary file bazel-out/host/bin/tensorflow/compiler/jit/kernels/_objs/xla_ops/xla_ops.pic.o matches
Binary file bazel-out/host/bin/tensorflow/compiler/jit/libcompilation_passes.pic.a matches
bazel-out/host/bin/tensorflow/compiler/jit/libcompilation_passes.pic.a-2.params:bazel-out/host/bin/tensorflow/compiler/jit/_objs/compilation_passes/build_xla_ops_pass.pic.o
Binary file bazel-out/host/bin/tensorflow/compiler/jit/_objs/compilation_passes/build_xla_ops_pass.pic.o matches
bazel-out/host/bin/tensorflow/compiler/jit/_objs/compilation_passes/build_xla_ops_pass.pic.d:bazel-out/host/bin/tensorflow/compiler/jit/_objs/compilation_passes/build_xla_ops_pass.pic.o: \
bazel-out/host/bin/tensorflow/compiler/jit/_objs/compilation_passes/build_xla_ops_pass.pic.d: tensorflow/compiler/jit/build_xla_ops_pass.cc /usr/include/stdc-predef.h \
bazel-out/host/bin/tensorflow/compiler/jit/_objs/compilation_passes/build_xla_ops_pass.pic.d: tensorflow/compiler/jit/build_xla_ops_pass.h \
bazel-out/host/bin/tensorflow/compiler/jit/_objs/jit_compilation_passes/jit_compilation_pass_registration.pic.d: /usr/include/stdc-predef.h tensorflow/compiler/jit/build_xla_ops_pass.h \
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/gen_gen_xla_ops_py_wrappers_cc matches
bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so-2.params:bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so matches
bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/gen_gen_xla_ops_py_wrappers_cc.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_gen_xla_ops_py_wrappers_cc /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/gen_gen_xla_ops_py_wrappers_cc
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/libxla_ops.pic.lo matches
bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/gen_gen_xla_ops_py_wrappers_cc.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_gen_xla_ops_py_wrappers_cc /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/gen_gen_xla_ops_py_wrappers_cc
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/libxla_ops.lo matches
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_objs/xla_ops/xla_ops.pic.o matches
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_objs/xla_ops/xla_ops.o matches
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o matches
bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.d:bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_objs/_xla_ops.so/xla_ops.pic.o: \
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/cc/ops/xla_jit_ops_gen_cc matches
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/cc/ops/xla_ops_gen_cc matches
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/cc/libxla_ops.pic.a matches
Binary file bazel-out/host/bin/tensorflow/compiler/tf2xla/cc/_objs/xla_ops/xla_ops.pic.o matches
bazel-out/host/bin/tensorflow/create_tensorflow.python_api_2_tf_python_api_gen_v2.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/create_tensorflow.python_api_2_tf_python_api_gen_v2.runfiles_manifest:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/host/bin/tensorflow/create_tensorflow.python_api_2_tf_python_api_gen_v2.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/_xla_ops.so /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so
bazel-out/host/bin/tensorflow/create_tensorflow.python_api_2_tf_python_api_gen_v2.runfiles/MANIFEST:org_tensorflow/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py /root/.cache/bazel/_bazel_root/74858e343c9a5ade9efa751b4acd0d13/execroot/org_tensorflow/bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py
bazel-out/host/genfiles/tensorflow/compiler/tf2xla/ops/gen_xla_ops.py:Original C++ source file: gen_xla_ops.cc
bazel-out/_tmp/action_outs/stderr-17882:                 from ./tensorflow/compiler/jit/build_xla_ops_pass.h:20,
bazel-out/_tmp/action_outs/stderr-17882:                 from tensorflow/compiler/jit/build_xla_ops_pass.cc:16:
bazel-out/_tmp/action_outs/stderr-17886:                 from ./tensorflow/compiler/jit/build_xla_ops_pass.h:20,

``

## cudart_static
```
# find bazel-out/ -type f |xargs grep "cudart_static"
bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so-2.params:bazel-out/k8-opt/genfiles/external/local_config_cuda/cuda/cuda/lib/libcudart_static.a
Binary file bazel-out/k8-opt/genfiles/external/local_config_cuda/cuda/cuda/lib/libcudart_static.a matches
bazel-out/host/bin/tensorflow/compiler/tf2xla/ops/_xla_ops.so-2.params:bazel-out/host/genfiles/external/local_config_cuda/cuda/cuda/lib/libcudart_static.a
Binary file bazel-out/host/genfiles/external/local_config_cuda/cuda/cuda/lib/libcudart_static.a matches
root@zhaojp-linux:/github/tf-src/tensorflow-1.14.0# 
root@zhaojp-linux:/github/tf-src/tensorflow-1.14.0# ll /usr/local/lib/python3.6/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so -h
-rwxr-xr-x 1 root staff 874M Oct  1 00:51 /usr/local/lib/python3.6/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so*
root@zhaojp-linux:/github/tf-src/tensorflow-1.14.0# ll bazel-out/host/bin/tensorflow/python/_pywrap_tensorflow_internal.so -h
-r-xr-xr-x 1 root root 381M Feb  2 07:27 bazel-out/host/bin/tensorflow/python/_pywrap_tensorflow_internal.so*
root@zhaojp-linux:/github/tf-src/tensorflow-1.14.0# nm -s bazel-out/host/bin/tensorflow/python/_pywrap_tensorflow_internal.so|grep cudaGetDevice
0000000013dacc00 b _ZGVZ13cudaGetDeviceE8func_ptr
0000000013dacc60 b _ZGVZ18cudaGetDeviceCountE8func_ptr
0000000013dacbd0 b _ZGVZ18cudaGetDeviceFlagsE8func_ptr
0000000013dacc50 b _ZGVZ23cudaGetDevicePropertiesE8func_ptr
0000000013dacc08 b _ZZ13cudaGetDeviceE8func_ptr
0000000013dacc68 b _ZZ18cudaGetDeviceCountE8func_ptr
0000000013dacbd8 b _ZZ18cudaGetDeviceFlagsE8func_ptr
0000000013dacc58 b _ZZ23cudaGetDevicePropertiesE8func_ptr
000000000a1a8a20 t cudaGetDevice
000000000a1a83f0 t cudaGetDeviceCount
000000000a1a8d10 t cudaGetDeviceFlags
000000000a1a8490 t cudaGetDeviceProperties
```
