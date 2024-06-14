import os, platform
from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
from PIL import Image
import random
import os

def color_boxes():
  def generate_light_color():
    """Generate a random light color."""
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    return (r, g, b)
  
  def generate_dark_color():
    """Generate a random dark color."""
    r = random.randint(0, 75)
    g = random.randint(0, 75)
    b = random.randint(0, 75)
    return (r, g, b)
  
  def generate_random_color():
    """Generate a random color."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)
  
  def generate_orange_color():
    """Generate orange color."""
    r = random.randint(200, 255)
    g = random.randint(100, 180)
    b = random.randint(0, 50)
    return (r, g, b)

  def generate_tree_green_color():
    """Generate a tree gree color."""
    r = random.randint(0, 40)
    g = random.randint(80, 150)
    b = random.randint(0, 30)
    return (r, g, b)

  def generate_road_black_color():
    """Generate a black color."""
    r = random.randint(0, 150)
    g = r
    b = r
    return (r, g, b)
  
  def generate_sky_blue_color():
    """Generate a random sky blue color."""
    r = random.randint(0, 50)
    g = random.randint(100, 230)
    b = random.randint(220, 255)
    return (r, g, b)

  def create_image(color, size=(160, 120)):
      """Create an image with the specified color and size."""
      image = Image.new('RGB', size, color)
      return image
  
  def create_random_image(size=(160, 120)):
    """Create an image with randomly colored pixels."""
    image = Image.new('RGB', size)
    pixels = image.load()
    
    for i in range(size[0]):
        for j in range(size[1]):
            color = generate_random_color()
            pixels[i, j] = color
    
    return image

  # Generate 400 concpet images
  for i in range(400):
      color = generate_road_black_color() #call
      image = create_image(color)
      image.save(f'.../dataset/data/tcav/image/concepts/.../{i+1}.jpg')

  print("Concepts generated!")


color_boxes()
