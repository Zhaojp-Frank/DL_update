# CUDA JIT sample
## general routine
- compile from ptx|src code to CUmodule, then get CUfunction from module
- JIT is driver API
```
# from ptx
 cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)ptx_source.c_str(),
                        strlen(ptx_source.c_str()) + 1, 0, 0, 0, 0);
# from src file?
cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
    unsigned int numOptions, CUjit_option *options, void **optionValues)

typedef enum CUjitInputType_enum
{
    /*** Compiled device-class-specific device code, Applicable options: none */
    CU_JIT_INPUT_CUBIN = 0,

    // PTX source code\n Applicable options: PTX compiler options
    CU_JIT_INPUT_PTX,

    // Bundle of multiple cubins and/or PTX of some device code\n     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
    CU_JIT_INPUT_FATBINARY,

    // Host object with embedded device code\n Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
    CU_JIT_INPUT_OBJECT,

    // Archive of host objects with embedded device code\n Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
    CU_JIT_INPUT_LIBRARY,
    CU_JIT_NUM_INPUT_TYPES
} CUjitInputType;

```
## how to compile from python src?
- also from ptx?
- tf: ./stream_executor/cuda/cuda_driver.cc

## sample
- in /usr/local/cuda/samples/6_Advanced/matrixMulDynlinkJIT
- or, /usr/local/cuda/samples/6_Advanced/6_Advanced/ptxjit/
```
# init dev: cuInit, cuCtxCreate
# prepare linker ,  set JIT opt: 
  // Setup linker options
  // Return walltime from JIT compilation
  options[0] = CU_JIT_WALL_TIME;
  optionVals[0] = (void *)&walltime;
  // Pass a buffer for info messages
  options[1] = CU_JIT_INFO_LOG_BUFFER;
  optionVals[1] = (void *)info_log;
  // Pass the size of the info buffer
  options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optionVals[2] = (void *)(long)logSize;
  // Pass a buffer for error message
  options[3] = CU_JIT_ERROR_LOG_BUFFER;
  optionVals[3] = (void *)error_log;
  // Pass the size of the error buffer
  options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optionVals[4] = (void *)(long)logSize;
  // Make the linker verbose
  options[5] = CU_JIT_LOG_VERBOSE;
  optionVals[5] = (void *)1;

  // Create a pending linker invocation
  checkCudaErrors(cuLinkCreate(6, options, optionVals, lState));

  // first search for the module path before we load the results
  if (!findModulePath(PTX_FILE, module_path, argv, ptx_source)) {
# link ptx
 // Load the PTX from the ptx file
  printf("Loading ptxjit_kernel[] program\n");
  myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)ptx_source.c_str(),
                        strlen(ptx_source.c_str()) + 1, 0, 0, 0, 0);
 // Complete the linker step
  checkCudaErrors(cuLinkComplete(*lState, &cuOut, &outSize));


# compile header etc into module
cuModuleLoadDataEx(&cuModule, matrixMul_kernel_32_ptxdump, jitNumOptions, jitOptions, (void **)jitOptVals);
 // Load resulting cuBin into module
  checkCudaErrors(cuModuleLoadData(phModule, cuOut));

# get function from module
CUfunction cuFunction;
cuModuleGetFunction(&cuFunction, cuModule, "mykernel");

# launch the kernel
      void *args[5] = { &d_C, &d_A, &d_B, &Matrix_Width_A, &Matrix_Width_B };
      checkCudaErrors(cuLaunchKernel(matrixMul, (WC/block_size), (HC/block_size), 1,
                                       block_size     , block_size     , 1,
                                       0,
                                       NULL, args, NULL));
#==============build== get ptx at first=--
nvcc -ccbin g++ -I../../common/inc -m64  -gencode arch=compute_30,code=compute_30 -o ptxjit.o -c ptxjit.cpp
nvcc -ccbin g++ -m64 -gencode arch=compute_30,code=compute_30 -o ptxjit ptxjit.o -lcuda -lcudart

```
