import os
from toolz.functoolz import pipe
import PIL.Image
import PIL.ImageOps
import PIL.ImageFilter
import scipy
import numpy as np
import skimage.filters
import skimage.transform

IM_SIZE = (400, 300)
IM_MIN = 0
IM_MAX = 255
SMOOTH_RAD = 2
#THRESH = 0.8

DATA_DIR = 'data'
PROCESSED_DIR = 'processed'
IMAGE_SUFFIXES = ('jpg', 'jpeg', 'png', 'bmp', 'gif')

# DEBUGGING

def inprint(arg):
    print(arg)
    return arg

# CONVERSION

def pil_2_numpy(image, dtype=np.float64):
    return (np.asarray(image.getdata(), dtype=dtype)
              .reshape((image.height, image.width)))

def numpy_2_pil(data):
    return PIL.Image.fromarray(np.uint8(
        data * 255 / (np.max(data) or 1)))

# PREPROCESSING

def preproc_resize(image):
    """Resize image."""
    image.thumbnail(IM_SIZE)
    return image

def preproc_grayscale(image):
    """RGB to grayscale."""
    return PIL.ImageOps.grayscale(image)

def preproc_rescale(image):
    """Rescale intensities."""
    return PIL.ImageOps.autocontrast(image)

def preproc_normalise(image):
    """Histogram normalisation."""
    return PIL.ImageOps.equalize(image)

def preproc_smooth(image):
    """Gaussian smoothing."""
    return image.filter(PIL.ImageFilter.GaussianBlur(SMOOTH_RAD))

# thresholding
def preproc_thresh(image):
    """Threshhold image."""
    pipeline = (
        pil_2_numpy,
        lambda i: i > skimage.filters.threshold_otsu(i),
        #lambda i: i > THRESH * 255,
        numpy_2_pil,
    )
    return pipe(image, *pipeline)

def preprocess(image):
    pipeline = (
        preproc_resize,
        preproc_grayscale,
        preproc_rescale,
        #preproc_normalise,
        preproc_smooth,
        preproc_thresh,
    )
    return pipe(image, *pipeline)

# FEATURE EXTRACTION

def extract_features(image):
    data = pil_2_numpy(image)
    hough, a, d = skimage.transform.hough_line(data)
    peaks, line_as, line_ds = skimage.transform.hough_line_peaks(
        hough, a, d)

    features = dict(
        hough_sd = np.std(hough.flat),
        hough_kurtosis = scipy.stats.kurtosis(hough.flat),
        hough_max_z = np.max(scipy.stats.mstats.zscore(hough.flat)),
        hough_line_sd = np.std(line_as),
        hough_no_lines = len(peaks),
    )
    return features

def classify_image(image):
    return extract_features(preprocess(PIL.Image.open(image)))
