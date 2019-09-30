import numpy as np
import cv2
from os import listdir
from os.path import join
from PIL import Image


def crop_numpy(dim1, dim2, dim3, vol):
    return vol[dim1:vol.shape[0] - dim1, dim2:vol.shape[1] - dim2, dim3:vol.shape[2] - dim3]


def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)


def get_dir(path):
    tiffs = [join(path, f) for f in listdir(path) if f[0] != '.']
    return sorted(tiffs)


def crop_cube(x, y, z, vol, cube_length=64):
    # Cube shape
    return crop_box(x, y, z, vol, (cube_length, cube_length, cube_length))


def crop_box(x, y, z, vol, shape):
    return vol[z:z + shape[2], x:x + shape[0], y:y + shape[1]]


def crop_cube_efficient(x, y, z, path, cube_length=64):
    vol = np.zeros((cube_length, cube_length, cube_length), dtype='uint16')

    fnames = get_dir(path)

    for i in range(cube_length):
        img = cv2.imread(fnames[z + i], cv2.COLOR_BGR2GRAY)
        vol[i] = img[y:y + cube_length, x:x + cube_length]

    return vol


def read_folder_section(path, start_index, end_index):
    fnames = get_dir(path)
    vol = []

    for f in fnames[start_index: end_index]:
        img = cv2.imread(f, cv2.COLOR_BGR2GRAY)
        vol.append(img)

    vol = np.array(vol)

    return vol


def read_folder_stack(path):
    fnames = get_dir(path)

    vol = cv2.imread(fnames[0], cv2.COLOR_BGR2GRAY)

    if len(vol.shape) == 3:
        print("Weird format")
        return vol

    vol = []

    for f in fnames:
        img = cv2.imread(f, cv2.COLOR_BGR2GRAY)
        vol.append(img)

    vol = np.array(vol)

    return vol


def read_tiff_stack(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)

    return np.array(images)


def coordinate_vol(coords, shape):
    vol = np.zeros(shape, dtype="uint16")
    for c in coords:
        vol[c[0], c[1], c[2]] = 1
    return vol


def preprocess(vol):
    return vol / 65535


def preprocess_batch(batch):
    assert len(batch.shape) == 5
    lst = []

    for i in range(batch.shape[0]):
        lst.append(preprocess(batch[i]))

    return np.array(lst)


def dist(p1, p2):
    sqr = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    return sqr ** .5

