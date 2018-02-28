#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

/*
layer是模型的关键，计算的基础单元。layer的操作有卷积、pool、内积、应用非线性elementwise变换（ReLU，sigmoid等），normalize，
加载数据，计算损失（softmax和hinge）等。
每个layer从底层连接获得输入，输出到顶层连接.

每个layer定义三种关键计算：setup，forward和backward。
setup：模型初始化时，初始化这个layer和它的连接；
forward：从底层取出输入，计算输出，传递到顶层；
backward：从顶层获得输入，计算梯度，发送到底层。有参数的层计算关于它的参数的梯度，并在内部存储。
特别地，每个layer有两种forward和backward实现，分别是基于CPU和GPU。如果没有实现GPU版本，layer默认使用CPU函数，这给快速实验带来了便利。

*/
template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

//// Instantiate a class with float and double specifications.
//#define INSTANTIATE_CLASS(classname) \
//    char gInstantiationGuard##classname; \
//    template class classname<float>; \
//    template class classname<double>

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
