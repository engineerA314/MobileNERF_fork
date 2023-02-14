import copy
import gc
import json
import os
import numpy
import cv2
from tqdm import tqdm
import pickle
import jax
import jax.numpy as np
from jax import random
import flax
import flax.linen as nn
import functools
import math
from typing import Sequence, Callable
import time
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing.pool import ThreadPool

# General math functions.

def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return np.matmul(a, b, precision=jax.lax.Precision.HIGHEST)

def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x, axis=-1, keepdims=True)

def sinusoidal_encoding(position, minimum_frequency_power,
    maximum_frequency_power,include_identity = False):
  # Compute the sinusoidal encoding components
  frequency = 2.0**np.arange(minimum_frequency_power, maximum_frequency_power)
  angle = position[..., None, :] * frequency[:, None]
  encoding = np.sin(np.stack([angle, angle + 0.5 * np.pi], axis=-2))
  # Flatten encoding dimensions
  encoding = encoding.reshape(*position.shape[:-1], -1)
  # Add identity component
  if include_identity:
    encoding = np.concatenate([position, encoding], axis=-1)
  return encoding

# Pose/ray math.

def generate_rays(pixel_coords, pix2cam, cam2world):
  """Generate camera rays from pixel coordinates and poses."""
  homog = np.ones_like(pixel_coords[..., :1])
  pixel_dirs = np.concatenate([pixel_coords + .5, homog], axis=-1)[..., None]
  cam_dirs = matmul(pix2cam, pixel_dirs)
  ray_dirs = matmul(cam2world[..., :3, :3], cam_dirs)[..., 0]
  ray_origins = np.broadcast_to(cam2world[..., :3, 3], ray_dirs.shape)

  #f = 1./pix2cam[0,0]
  #w = -2. * f * pix2cam[0,2]
  #h =  2. * f * pix2cam[1,2]

  return ray_origins, ray_dirs

def pix2cam_matrix(height, width, focal):
  """Inverse intrinsic matrix for a pinhole camera."""
  return  np.array([
      [1./focal, 0, -.5 * width / focal],
      [0, -1./focal, .5 * height / focal],
      [0, 0, -1.],
  ])

def camera_ray_batch(cam2world, hwf):
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  height, width = int(hwf[0]), int(hwf[1])
  pix2cam = pix2cam_matrix(*hwf)
  pixel_coords = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
  return generate_rays(pixel_coords, pix2cam, cam2world)

def camera_ray_batch_stage2(cam2world, hwf): ### antialiasing by supersampling
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  height, width = int(hwf[0]), int(hwf[1])
  pix2cam = pix2cam_matrix(*hwf)
  x_ind, y_ind = np.meshgrid(np.arange(width), np.arange(height))
  pixel_coords = np.stack([x_ind-0.25, y_ind-0.25, x_ind+0.25, y_ind-0.25,
                  x_ind-0.25, y_ind+0.25, x_ind+0.25, y_ind+0.25], axis=-1)
  pixel_coords = np.reshape(pixel_coords, [height,width,4,2])

  return generate_rays(pixel_coords, pix2cam, cam2world)

def camera_ray_batch_stage3b(cam2world, hwf): ### antialiasing by supersampling
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  height, width = int(hwf[0]), int(hwf[1])
  pix2cam = pix2cam_matrix(*hwf)
  x_ind, y_ind = np.meshgrid(np.arange(width), np.arange(height))
  pixel_coords = np.stack([x_ind-0.25, y_ind-0.25, x_ind+0.25, y_ind-0.25,
                  x_ind-0.25, y_ind+0.25, x_ind+0.25, y_ind+0.25], axis=-1)
  pixel_coords = np.reshape(pixel_coords, [height,width,4,2])

  return generate_rays(pixel_coords, pix2cam, cam2world)

def camera_ray_batch_stage3(cam2world, hwf): ### antialiasing by supersampling
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  height, width = int(hwf[0]), int(hwf[1])
  pix2cam = pix2cam_matrix(*hwf)
  x_ind, y_ind = np.meshgrid(np.arange(width), np.arange(height))
  pixel_coords = np.stack([x_ind-0.25, y_ind-0.25, x_ind+0.25, y_ind-0.25,
                  x_ind-0.25, y_ind+0.25, x_ind+0.25, y_ind+0.25], axis=-1)
  pixel_coords = np.reshape(pixel_coords, [height,width,4,2])

  return generate_rays(pixel_coords, pix2cam, cam2world)


def random_ray_batch(rng, batch_size, data):
  """Generate a random batch of ray data."""
  keys = random.split(rng, 3)
  cam_ind = random.randint(keys[0], [batch_size], 0, data['c2w'].shape[0])
  y_ind = random.randint(keys[1], [batch_size], 0, data['images'].shape[1])
  x_ind = random.randint(keys[2], [batch_size], 0, data['images'].shape[2])
  pixel_coords = np.stack([x_ind, y_ind], axis=-1)
  pix2cam = pix2cam_matrix(*data['hwf'])
  cam2world = data['c2w'][cam_ind, :3, :4]
  rays = generate_rays(pixel_coords, pix2cam, cam2world)
  pixels = data['images'][cam_ind, y_ind, x_ind]
  return rays, pixels

def random_ray_batch_stage2(rng, batch_size, data): ### antialiasing by supersampling
  """Generate a random batch of ray data."""
  keys = random.split(rng, 3)
  cam_ind = random.randint(keys[0], [batch_size], 0, data['c2w'].shape[0])
  y_ind = random.randint(keys[1], [batch_size], 0, data['images'].shape[1])
  y_ind_f = y_ind.astype(np.float32)
  x_ind = random.randint(keys[2], [batch_size], 0, data['images'].shape[2])
  x_ind_f = x_ind.astype(np.float32)
  pixel_coords = np.stack([x_ind_f-0.25, y_ind_f-0.25, x_ind_f+0.25, y_ind_f-0.25,
                  x_ind_f-0.25, y_ind_f+0.25, x_ind_f+0.25, y_ind_f+0.25], axis=-1)
  pixel_coords = np.reshape(pixel_coords, [batch_size,4,2])
  pix2cam = pix2cam_matrix(*data['hwf'])
  cam_ind_x4 = np.tile(cam_ind[..., None], [1,4])
  cam_ind_x4 = np.reshape(cam_ind_x4, [-1])
  cam2world = data['c2w'][cam_ind_x4, :3, :4]
  cam2world = np.reshape(cam2world, [batch_size,4,3,4])
  rays = generate_rays(pixel_coords, pix2cam, cam2world)
  pixels = data['images'][cam_ind, y_ind, x_ind]
  return rays, pixels

def random_ray_batch_stage3b(rng, batch_size, data): ### antialiasing by supersampling
  """Generate a random batch of ray data."""
  keys = random.split(rng, 3)
  cam_ind = random.randint(keys[0], [batch_size], 0, data['c2w'].shape[0])
  y_ind = random.randint(keys[1], [batch_size], 0, data['images'].shape[1])
  y_ind_f = y_ind.astype(np.float32)
  x_ind = random.randint(keys[2], [batch_size], 0, data['images'].shape[2])
  x_ind_f = x_ind.astype(np.float32)
  pixel_coords = np.stack([x_ind_f-0.25, y_ind_f-0.25, x_ind_f+0.25, y_ind_f-0.25,
                  x_ind_f-0.25, y_ind_f+0.25, x_ind_f+0.25, y_ind_f+0.25], axis=-1)
  pixel_coords = np.reshape(pixel_coords, [batch_size,4,2])
  pix2cam = pix2cam_matrix(*data['hwf'])
  cam_ind_x4 = np.tile(cam_ind[..., None], [1,4])
  cam_ind_x4 = np.reshape(cam_ind_x4, [-1])
  cam2world = data['c2w'][cam_ind_x4, :3, :4]
  cam2world = np.reshape(cam2world, [batch_size,4,3,4])
  rays = generate_rays(pixel_coords, pix2cam, cam2world)
  pixels = data['images'][cam_ind, y_ind, x_ind]
  return rays, pixels

def random_ray_batch_stage3(rng, batch_size, data): ### antialiasing by supersampling
  """Generate a random batch of ray data."""
  keys = random.split(rng, 3)
  cam_ind = random.randint(keys[0], [batch_size], 0, data['c2w'].shape[0])
  y_ind = random.randint(keys[1], [batch_size], 0, data['images'].shape[1])
  y_ind_f = y_ind.astype(np.float32)
  x_ind = random.randint(keys[2], [batch_size], 0, data['images'].shape[2])
  x_ind_f = x_ind.astype(np.float32)
  pixel_coords = np.stack([x_ind_f-0.25, y_ind_f-0.25, x_ind_f+0.25, y_ind_f-0.25,
                  x_ind_f-0.25, y_ind_f+0.25, x_ind_f+0.25, y_ind_f+0.25], axis=-1)
  pixel_coords = np.reshape(pixel_coords, [batch_size,4,2])
  pix2cam = pix2cam_matrix(*data['hwf'])
  cam_ind_x4 = np.tile(cam_ind[..., None], [1,4])
  cam_ind_x4 = np.reshape(cam_ind_x4, [-1])
  cam2world = data['c2w'][cam_ind_x4, :3, :4]
  cam2world = np.reshape(cam2world, [batch_size,4,3,4])
  rays = generate_rays(pixel_coords, pix2cam, cam2world)
  pixels = data['images'][cam_ind, y_ind, x_ind]
  return rays, pixels


# Learning rate helpers.

def log_lerp(t, v0, v1):
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0:
    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = np.log(v0)
  lv1 = np.log(v1)
  return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)

def lr_fn(step, max_steps, lr0, lr1, lr_delay_steps=20000, lr_delay_mult=0.1):
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  return delay_rate * log_lerp(step / max_steps, lr0, lr1)

def lr_fn(step, max_steps, lr0, lr1, lr_delay_steps=20000, lr_delay_mult=0.1):
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  return delay_rate * log_lerp(step / max_steps, lr0, lr1)
