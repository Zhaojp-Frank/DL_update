# TF-xla
## how to enable
- TF_XLA_FLAGS=--tf_xla_auto_jit=2 python ... 
- or, tf.config.optimizer.set_jit(True)
- to enable on CPU: TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" python ..

## dump
- XLA_FLAGS="--dump_hlo_as_text --xla_dump_to=/tmp/tf-xla" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python ... 
-- then goto /tmp/tf-xla
- dump graph: TF_DUMP_GRAPH_PREFIX=/tmp/tf-xla TF_XLA_FLAGS="--tf_xla_clustering_debug"
