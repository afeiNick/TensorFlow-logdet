# TensorFlow-logdet
batched logdet GPU op for tensorflow

The following bash commands have been used to compile the op on a Linux machine:

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

nvcc -std=c++11 -c -o myLogdet_op.cu.o myLogdet_op_gpu_cublas.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

gcc -std=c++11 -shared -o myLogdet_op.so myLogdet_op.cc myLogdet_op.cu.o -I $TF_INC -fPIC -L/usr/local/cuda/lib64/ -lcublas -lcudart

rm *.o
