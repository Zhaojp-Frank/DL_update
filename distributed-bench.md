# TF distributed CNN benchmark script
## lightwieght CNN networks, 
- e.g., mobilenet, shufflenet, SSD etc

## script to bench CNN
- TF 1.13(+)
```
#!/bin/bash
#models=(alexnet vgg16 vgg19 lenet googlenet overfeat trivial inception3 inception4 resnet50 resnet50_v2 resnet101 resnet101_v2 nasnet mobilenet)
models=(mobilenet alexnet vgg16 resnet50 inception3 nasnet)
dataset="--data_name imagenet"
dataDir='--data_dir=/opt/imagenet/'

step="--num_batches 1000"
warmup="--num_warmup_batches 10"
batchSz=(64 128 256 512)
gpuN=(1 2 4 8)

#parameter_server replicated
# distributed_replicated : PS
# distributed_all_reduce :
# collective_all_reduce  : diff with distributed all reduce?
# horovod : MPI, NCCL/MPI all-reduce
#independent
replica="--variable_update replicated"
replicaPS="--variable_update parameter_server"
replicaFree="--variable_update independent"
psCPU='--local_parameter_device=cpu'

xla="--xla True --xla_compile True"

FP16='--use_fp16=True'
inf='--forward_only=True'
freezeGraph="--freeze_when_forward_only True"
trt="--trt_mode FP16"

useUMA="--use_unified_memory True"
gmempct="--gpu_memory_frac_for_testing 100"

graphLog="--graph_file ./graph-1.txt"
tfprofLog="--tfprof_file ./tfprof-1.log"
maxOpTrack="--gpu_kt_max_interval -"
trainAcc="--print_training_accuracy True"

mlPerf="--ml_perf True"
useDataset="--use_datasets True"
winogradOpt="--winograd_nonfused True"
batchnormPersistent="--batchnorm_persistent True"
compactGrad="--compact_gradient_transfer True"
splitGrad="--gradient_repacking 2"
relaxSyncVar="--variable_consistency relaxed"
cacheData="--datasets_repeat_cached_sample True"

#nvp='nvprof --analysis-metrics --concurrent-kernels on --continuous-sampling-interval 1 --dependency-analysis --devices 0 --kernel-latency-timestamps on --profile-all-processes -o %p.nvvp '
#nvp='nvprof --replay-mode kernel --metrics flop_sp_efficiency --skip-kernel-replay-save-restore on --profile-api-trace all --analysis-metrics --concurrent-kernels on --continuous-sampling-interval 2 --dependency-analysis --devices 0 --kernel-latency-timestamps off --profile-child-processes -o %p.nvvp '

for model in ${models[@]}; do
  for n in ${gpuN[@]}; do
  for b in ${batchSz[@]}; do
    #LD_PRELOAD=/opt/bench/libxinit.so.1 XG=0 shmNow=1 LD_LIBRARY_PATH=/opt/bench/:$LD_LIBRARY_PATH \
    if [ $n -eq 1 ]; then
      python ./tf_cnn_benchmarks.py --num_gpus $n $dataset --model $model --batch_size $b $step ; sleep 5
    else
      python ./tf_cnn_benchmarks.py --num_gpus $n $dataset --model $model --batch_size $b $step $replicaPS; sleep 5
      python ./tf_cnn_benchmarks.py --num_gpus $n $dataset --model $model --batch_size $b $step $replica; sleep 5
      python ./tf_cnn_benchmarks.py --num_gpus $n $dataset --model $model --batch_size $b $step $replicaFree;sleep 5
     fi
  done
  done
done
```

## script to monitor GPU activites, power, PCIe
```
#!/bin/bash
dev=$1
cmd="nvidia-smi dmon -i ${dev} -s put"
for (( i=0; i<14400; i++ ));do
        $cmd
        sleep 1
done
```

