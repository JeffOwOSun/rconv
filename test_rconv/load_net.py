import os
import caffe

if __name__ == '__main__':
    caffe.set_mode_cpu()
    # caffe.set_device(0)

    lenet_ptx = os.path.join(os.path.dirname(__file__), 'lenet.prototxt')
    lenet_ptx = str(lenet_ptx) # what the hack is happening here?
    net = caffe.Net(lenet_ptx,caffe.TEST)

