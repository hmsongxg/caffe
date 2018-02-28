#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

/*
layer��ģ�͵Ĺؼ�������Ļ�����Ԫ��layer�Ĳ����о����pool���ڻ���Ӧ�÷�����elementwise�任��ReLU��sigmoid�ȣ���normalize��
�������ݣ�������ʧ��softmax��hinge���ȡ�
ÿ��layer�ӵײ����ӻ�����룬�������������.

ÿ��layer�������ֹؼ����㣺setup��forward��backward��
setup��ģ�ͳ�ʼ��ʱ����ʼ�����layer���������ӣ�
forward���ӵײ�ȡ�����룬������������ݵ����㣻
backward���Ӷ��������룬�����ݶȣ����͵��ײ㡣�в����Ĳ����������Ĳ������ݶȣ������ڲ��洢��
�ر�أ�ÿ��layer������forward��backwardʵ�֣��ֱ��ǻ���CPU��GPU�����û��ʵ��GPU�汾��layerĬ��ʹ��CPU�������������ʵ������˱�����

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
