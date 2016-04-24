#!/usr/bin/env python3

import os
import csv
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

def apply(f, x):
    return f(x)

def inprint(arg):
    print(arg)
    return arg

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

def test_preprocess():
    test_im_fn = '1.jpg'
    test_im = PIL.Image.open(test_im_fn)
    pipeline = (
        preproc_resize,    # 2
        preproc_grayscale, # 3
        preproc_rescale,   # 4
        #preproc_normalise, # 5
        preproc_smooth,    # 6
        preproc_thresh,    # 7
    )
    this_pipeline = []
    for i, p in enumerate(pipeline):
        this_pipeline.append(p)
        pipe(test_im, *this_pipeline).save('{}.jpg'.format(i+2))

def preprocess_data_dir():
    for path, _, files in os.walk(DATA_DIR):
        out_dir  = path.replace(DATA_DIR, PROCESSED_DIR)

        os.makedirs(out_dir, exist_ok=True)
        im_files = (f for f in files
                   if f.lower().endswith(IMAGE_SUFFIXES))
        for im_file in im_files:
            unprocessed = PIL.Image.open(os.path.join(path, im_file))
            processed = preprocess(unprocessed)
            processed.save(os.path.join(out_dir, im_file))

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

def process_data_dir():
    datasets = (d for d in os.listdir(PROCESSED_DIR)
                if os.path.isdir(os.path.join(PROCESSED_DIR, d)))
    print(list(os.listdir(PROCESSED_DIR)))
    for path in datasets:
        im_files = (os.path.join(PROCESSED_DIR, path, f)
                    for f in os.listdir(os.path.join(PROCESSED_DIR, path))
                    if f.lower().endswith(IMAGE_SUFFIXES))

        dataset = [extract_features(image = PIL.Image.open(f))
                  for f in im_files]

        with open(os.path.join(PROCESSED_DIR, path + '.csv'), 'w+') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(dataset[0].keys()),
            )
            writer.writeheader()
            for d in dataset:
                writer.writerow(d)


if __name__ == '__main__':
    #test_preprocess()
    #preprocess_data_dir()
    process_data_dir()
