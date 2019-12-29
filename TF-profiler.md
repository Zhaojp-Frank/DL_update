# Profiling
## TF profiler
- 支持cpu/gpu, 类型包括floats, mem, param bytes, run time; 可通过工具有多种view(graph-timeline, scope/node, code, op), 以及timeline 输出到chrome浏览器可视化；可给出建议  
-- 筛选项: [name | depth|bytes|peak_bytes|residual_bytes|output_bytes|micros|accelerator_micros|cpu_micros|params|float_ops|occurrence]

- **不足**：离线分析；不支持并发多job；不支持分析传输-同步；需要改代码;profiler命令行默认不会build，自己build很麻烦。只适用于TF，不通用；且依赖一些op的实现以及用户的交互（例如有些op需要用户输入shape）  
-- e.g : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_memory.md
```
It must have RegisterStatistics('flops') defined in TensorFlow. tfprof uses the definition to calculate float operations. Contributions are welcomed.

It must have known "shape" information for RegisterStatistics('flops') to calculate the statistics. It is suggested to pass in -run_meta_path if shape is only known during runtime. 
```
## steps
### change code and add profiler
```
# User can control the tracing steps and
# dumping steps. User can also run online profiling during training.
#
# Create options to profile time/memory as well as parameters.
builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory()).order_by('micros').build()
opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()

# Collect traces of steps 10~20, dump the whole profile (with traces of
# step 10~20) at step 20. The dumped profile can be used for further profiling
# with command line interface or Web UI.
with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                      trace_steps=range(10, 20),
                                      dump_steps=[20]) as pctx:
  # Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.
  pctx.add_auto_profiling('op', opts, [15, 18, 20])
  # Run online profiling with 'scope' view and 'opts2' options at step 20.
  pctx.add_auto_profiling('scope', opts2, [20])
  # High level API, such as slim, Estimator, etc.
  train_loop()
```
### Run the code
```
bazel-bin/tensorflow/core/profiler/profiler \
    --profile_path=/tmp/train_dir/profile_xx
tfprof> op -select micros,bytes,occurrence -order_by micros

# Profiler ui available at: https://github.com/tensorflow/profiler-ui
python ui.py --profile_context_path=/tmp/train_dir/profile_xx
```
### Check the stat
```
# The following example generates a timeline.
tfprof> graph -step -1 -max_depth 100000 -output timeline:outfile=<filename>

generating trace file.

******************************************************
Timeline file is written to <filename>.
Open a Chrome browser, enter URL chrome://tracing and load the timeline file.
******************************************************
# In code view, the time of each line of Python code is the aggregated
# times of all operations created by that line.
# In command line, it requires --graph_path --op_log_path and --run_meta_path.
# --op_log_path provides the code traces information.
# --run_meta_path provides the time information.
tfprof> code -show_name_regexes seq2seq_attention.* -max_depth 10 -select micros -order_by micros

# Sometimes you want to explore a specific function. You can do that
# with -start_name_regexes.
tfprof> code -start_name_regexes .*_add_seq2seq.* -show_name_regexes seq2seq_attention.* -max_depth 10 -select micros -order_by micros
node name | execution time

# You can also dive deeper into tensorflow's libraries.
tfprof> code  -max_depth 5 -select micros -order_by micros -start_name_regexes .*_add_seq2seq.* -min_micros 100000

# In op view, you can view the aggregated time of each operation type.
tfprof> op -select micros,occurrence -order_by micros

# You might be surprised to see that SoftmaxCrossEntropyWithLogits is
# that expensive. As shown below, it is placed on cpu.
tfprof> op -select micros,device -order_by micros

#Usually scope view allows you to pin point the problematic places if you have properly named your operations with tf.name_scope or tf.variable_scope.
tfprof> scope -max_depth 30 -select micros -min_micros 100000 -order_by micros
node name | execution time

# 与代码对齐
tfprof> code -max_depth 1000 -show_name_regexes .*model_analyzer.*py.* -select micros -account_type_regexes .* -order_by micros
# 参数量
tfprof> scope -account_type_regexes VariableV2 -max_depth 4 -select params
# 最慢的op
tfprof> op -select micros,bytes,occurrence -order_by micros
# To profile float operations in commandline, you need to pass --graph_path
# and --op_log_path.
tfprof> scope -min_float_ops 1 -select float_ops -account_displayed_op_only
node name | # float_ops
_TFProfRoot (--/17.63b flops)
  gradients/pool_logit/xw_plus_b/MatMul_grad/MatMul (163.84k/163.84k flops)
  gradients/pool_logit/xw_plus_b/MatMul_grad/MatMul_1 (163.84k/163.84k flops)
  init/init_conv/Conv2D (113.25m/113.25m flops)
  
- 给点建议
tfprof> advise
```

## example
- time python mnist_profile.py --steps 100 --out 100
```
# /usr/local/lib/python3.6/dist-packages/tensorflow/examples/tutorials/mnist
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Profiler is created here.
        profiler = tf.profiler.Profiler(sess.graph)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # Train
        n_steps = FLAGS.steps
        for i in range(n_steps):
            batch_xs, batch_ys = mnist.train.next_batch(100)

            run_metadata = tf.RunMetadata()
            sess.run(train_step, options=options, run_metadata=run_metadata, feed_dict={x: batch_xs, y_: batch_ys})
            # We collect profiling infos for each step.
            profiler.add_step(i, run_metadata)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        run_metadata = tf.RunMetadata()
        print(sess.run(accuracy, options=options, run_metadata=run_metadata, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        # I collect profiling infos for last step, too.
        profiler.add_step(n_steps, run_metadata)

        option_builder = tf.profiler.ProfileOptionBuilder
        opts = (option_builder(option_builder.time_and_memory()).
                with_step(-1). # with -1, should compute the average of all registered steps.
                with_file_output('test-%s.txt' % FLAGS.out).
                select(['micros','bytes','occurrence']).order_by('micros').
                build())
        # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
        profiler.profile_operations(options=opts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps to run.')
    parser.add_argument('--out', type=str, required=True, help='Output filename.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
 ```
## 通用的、实时的分析框架
- TF profiler问题
-- 通用性欠缺：算法开发者熟悉模型，谈不了解系统；系统开发-调优者很难应对各种模型、框架，最好有个通用的、运行时的profiling
-- 功能欠缺：tf-profiler无法同时profiler多个并发job。see: https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
- 方案：提供透明、通用的profiler，系统调优者在不改变现有模型、框架代码情况下，启动profiling，快速获得gpu前、中、后的性能数据；支持多job

# Ref  
1. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/options.md  
2. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md  
3. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_time.md  
4. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md  
2. https://jiayiliu.github.io/posts/tensoflow-profiling/  
3. https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras  
4. https://gist.github.com/notoraptor/4cfeaaf2ab24ebce59ac727f389096fa  
5. https://stackoverflow.com/questions/54360762/how-to-profile-tensorflow-model-that-running-on-tf-serving  
