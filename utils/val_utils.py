
import time
import numpy as np
from skimage.metrics import structural_similarity

def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return

def _as_floats_riandint(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    image0 = np.asarray(image0, dtype=np.float64)
    image1 = np.asarray(image1, dtype=np.float64)
    return image0, image1

def mean_squared_error_riandint(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats_riandint(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def peak_signal_noise_ratio_riandint(image_true, image_test, *, data_range=255):

    check_shape_equality(image_true, image_test)
    image_true, image_test = _as_floats_riandint(image_true, image_test)
    err = mean_squared_error_riandint(image_true, image_test)
    return 10 * np.log10((data_range ** 2) / err)

class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(restored, clean):
    assert restored.shape == clean.shape
    restored = np.expand_dims(restored, axis=0)
    clean = np.expand_dims(clean, axis=0)

    psnr = 0
    ssim = 0

    for i in range(restored.shape[0]):
        psnr += peak_signal_noise_ratio_riandint(clean[i], restored[i], data_range=255)
        ssim += structural_similarity(clean[i], restored[i], data_range=255, channel_axis=-1)

    return psnr / restored.shape[0], ssim / restored.shape[0], restored.shape[0]

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0