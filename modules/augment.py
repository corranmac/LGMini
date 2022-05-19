#@title Random Crop Augmentation Code from [@Grump_AI](https://twitter.com/Grump_AI).
import random
import math
from PIL import Image, ImageSequence, ImageOps, ImageEnhance, ImageFilter

def all_crops(image_path, distance_between_crops, out_path):
  """
  Saves all square-sized crops starting from the leftmost or topmost edge of the image.
  :param str image_path: Path to the image to crop
  :param int distance_between_crops: Distance between each unique crop in pixels
  :param str out_path: Directory to save images
  """

  image_name = image_path.split('/')[-1].split('.')[0]
  image = Image.open(image_path)

  # calculate valid center points
  width, height = image.size
  size = 256
  center_width = 0
  center_height = 0
  num_crops = 1
  horizontal = True
  if width > height:
    center_width = center_height = int(height/2)
    size = height
    num_crops = int(round((width-height)/distance_between_crops)) + 1
  elif height >= width:
    center_width = center_height = int(width/2)
    size = width
    num_crops = int(round((height-width)/distance_between_crops)) + 1
    horizontal = False
  s2 = int(round(size/2))
  
  for _ in range(num_crops):
    image.crop((center_width-s2, center_height-s2, center_width+s2, center_height+s2)).save(f'{out_path}/{image_name}-cropped-{size}px-{center_width}-{center_height}.png')
    if horizontal:
      center_width = center_width + distance_between_crops
    else:
      center_height = center_height + distance_between_crops
  print(str(num_crops) + " images generated line crop")
  
def random_crop(image_path, size, num_crops, out_path):
  """
  Save images cropped to desired size with random center points
  Only valid unique crops are saved, even if its fewer than num_crops
  :param str image_path: Path to the image to crop
  :param int size: Size in pixels of desired crop
  :param int num_crops: Number of crops desired
  :param str out_path: Directory to save images
  """

  image_name = image_path.split('/')[-1].split('.')[0]
  image = Image.open(image_path)

  if size % 2 != 0:
    raise Exception('crop size must be even')
  if any(map(lambda x: x < size+1, image.size)):
    raise Exception(f'image size of:{image.size} too small for crop size of: {size}')

  # calculate valid center points
  width, height = image.size
  left_bound, top_bound = size/2, size/2
  right_bound = width - size/2
  bottom_bound = height - size/2
  valid_center_points = int((right_bound - left_bound) * (bottom_bound - top_bound))

  # if there are fewer valid center points than requested crops
  # then produce fewer crops
  if valid_center_points < num_crops:
    num_crops = valid_center_points

  used_center_points = []
  for _ in range(num_crops):
    # ensure uniqueness of center points
    # slower when valid_center_points is near num_crops
    while True:
      w = random.randrange(left_bound, right_bound)
      h = random.randrange(top_bound, bottom_bound)
      if not ([w,h] in used_center_points):
        break

    used_center_points.append([w,h])

    image.crop((w-(size/2), h-(size/2), w+(size/2), h+(size/2))).save(f'{out_path}/{image_name}-cropped-{size}px-{w}-{h}.png')
  print(str(num_crops) + " images generated random crop")

def random_rotate(image_path, min, max, expand):
  theta = int(random.triangular(min, max))
  image = Image.open(image_path)
  image = image.rotate(theta, expand=expand).save(image_path)

def random_flip(image_path, horiz, verti):
  image = Image.open(image_path)
  if random.random() < horiz:
    image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
  if random.random() < verti:
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
  image.save(image_path)

def random_grayscale(image_path, chance):
  image = Image.open(image_path)
  if random.random() < chance:
    image = ImageOps.grayscale(image)
  image.save(image_path)

def random_invert(image_path, chance):
  image = Image.open(image_path)
  if random.random() < chance:
    image = ImageOps.invert(image)
  image.save(image_path)

def random_brightness(image_path, min, max):  
  image = Image.open(image_path)
  enhancer = ImageEnhance.Brightness(image)
  factor = random.triangular(min, max, 1)
  image = enhancer.enhance(factor)
  image.save(image_path)

def random_contrast(image_path, min, max):
  image = Image.open(image_path)
  enhancer = ImageEnhance.Contrast(image)
  factor = random.triangular(min, max, 1)
  image = enhancer.enhance(factor)
  image.save(image_path)

def random_saturation(image_path, min, max):
  image = Image.open(image_path)
  enhancer = ImageEnhance.Color(image)
  factor = random.triangular(min, max, 1)
  image = enhancer.enhance(factor)
  image.save(image_path)

def random_sharpness(image_path, min, max):
  image = Image.open(image_path)
  enhancer = ImageEnhance.Sharpness(image)
  factor = random.triangular(min, max, 1)
  image = enhancer.enhance(factor)
  image.save(image_path)

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = numpy.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = numpy.max(rgb[..., :3], axis=-1)
    minc = numpy.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = numpy.zeros_like(r)
    gc = numpy.zeros_like(g)
    bc = numpy.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = numpy.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = numpy.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = numpy.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = numpy.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = numpy.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def hueChange(img, hue):
    arr = numpy.array(img)
    hsv = rgb_to_hsv(arr)
    hsv[..., 0] = hue
    rgb = hsv_to_rgb(hsv)
    return Image.fromarray(rgb, 'RGB')

def hueShift(img, amount):
    arr = numpy.array(img)
    hsv = rgb_to_hsv(arr)
    hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
    rgb = hsv_to_rgb(hsv)
    return Image.fromarray(rgb, 'RGB')

def random_hueshift(image_path, min, max):
    image = Image.open(image_path).convert('RGB')
    amount = random.randint(min, max)
    image = hueShift(image, amount/360.)
    image.save(image_path)
