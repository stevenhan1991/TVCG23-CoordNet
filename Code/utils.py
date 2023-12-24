import os
import numpy as np
from skimage.transform import resize

def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[2]:s, :sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[3]:s, :sidelen[2]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    print(pixel_coords.shape)
    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    print(pixel_coords)
    return pixel_coords