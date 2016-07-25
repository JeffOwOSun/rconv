import caffe
import cv2
import numpy as np
import os

class DataLayer(caffe.Layer):
    """ doc """

    def setup(self, bottom, top):
        self._name_to_top_map = {'data': 0, 'frame_info': 1}
        self.frame_idx = 0

        data = self._compute_next_frame_blob()
        top[0].reshape(*(data.shape))
        top[1].reshape(1,)
        top[2].reshape(1,)
        print '>>>>Setting Up'

    def forward(self, bottom, top):
        data, idx = self._get_next_frame()

        top[0].reshape(*(data.shape))
        top[0].data[...] = data.astype(np.float32, copy=False)

        # frame_info is np.array([frame_index])
        frame_info = np.array([idx])
        top[1].reshape(*(frame_info.shape))
        top[1].data[...] = frame_info.astype(np.float32, copy=False)

        top[2].data[...] = np.array([5])

    def backward(self, top, propagat_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happends during the call to forward."""
        pass

    def _compute_next_frame_blob(self):
        frame_path = os.path.join(os.path.dirname(__file__), 'frame0.jpg')
        im = cv2.imread(frame_path)
        blob = np.array([im])
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def _get_next_frame(self):
        blob = self._compute_next_frame_blob()
        idx = self.frame_idx
        self.frame_idx += 1
        return blob, idx
