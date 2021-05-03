from scipy import ndimage
import cv2
import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from collections import defaultdict

def gaussian_blur(size=5, sigma=1.4):

    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def gradient_calculation(img):


    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)


    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255


    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def non_maximum_suppression(G, theta):

    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                if (G[i,j] >= q) and (G[i,j] >= r):
                    Z[i,j] = G[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

def double_thresholding(Z, lowThresholdRatio=0.09, highThresholdRatio=0.17, w_p_v=75):

    highThreshold = Z.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = Z.shape
    result = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(w_p_v)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= highThreshold)
    zeros_i, zeros_j = np.where(Z < lowThreshold)

    weak_i, weak_j = np.where((Z <= highThreshold) & (Z >= lowThreshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return (result, weak, strong)

def edge_tracking(img, weak, strong=255):

    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):

            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def canny(img, size, sigma, l_t_r, h_t_r, w_p_v):

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    S_I = convolve(gray, gaussian_blur(size, sigma))

    G, theta = gradient_calculation(S_I)

    Z = non_maximum_suppression(G, theta)

    result, weak, strong = double_thresholding(Z, l_t_r, h_t_r, w_p_v)

    img = edge_tracking(result, weak, strong)

    return img

def visualize(img, format=None, gray=False):

    if img.shape[0] == 3:
        img = img.transpose(1,2,0)

    plt.imshow(img, format)
    plt.show()

def line_detection(name, image, edge_image, num_rhos=180, num_thetas=180, t_count=220):

  I = Image.open(name)
  output_image = Image.new("RGB", I.size)
  draw_result = ImageDraw.Draw(output_image)

  edge_height, edge_width = edge_image.shape[:2]
  edge_height_half, edge_width_half = edge_height / 2, edge_width / 2

  d = np.sqrt(np.square(edge_height) + np.square(edge_width))
  dtheta = 180 / num_thetas
  drho = (2 * d) / num_rhos

  thetas = np.arange(0, 180, step=dtheta)
  rhos = np.arange(-d, d, step=drho)

  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))

  accumulator = np.zeros((len(rhos), len(rhos)))



  for y in range(edge_height):
    for x in range(edge_width):
      if edge_image[y][x] != 0:
        edge_point = [y - edge_height_half, x - edge_width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          rho_idx = np.argmin(np.abs(rhos - rho))
          accumulator[rho_idx][theta_idx] += 1
          ys.append(rho)
          xs.append(theta)


  for y in range(accumulator.shape[0]):
    for x in range(accumulator.shape[1]):
      if accumulator[y][x] > t_count:
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        draw_result.line([(x1, y1), (x2, y2)], fill=(255,255,255,255))

  output_image.save("output_lines.png")


  return accumulator, rhos, thetas

def filter_strong_edges(gradient, width, height, low, high):

    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))


    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)

def circle_detection(name, E_I, rmin, rmax, threshold):

    I = Image.open(name)
    output_image = Image.new("RGB", I.size)
    draw_result = ImageDraw.Draw(output_image)

    
    steps = 100
    

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    keep = filter_strong_edges(E_I, np.shape(I)[0], np.shape(I)[1], 20, 25)

    acc = defaultdict(int)
    for x, y in keep:
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            
            circles.append((x, y, r))

    for x, y, r in circles:
        draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,255,255,255))

    output_image.save("output_circles.png")



name = input("Enter input image name: ")
size = int(input("Enter kernel size: "))
sigma = float(input("Enter sigma value: "))
l_t_r = float(input("Enter low threshold ratio: "))
h_t_r = float(input("Enter high threshold ratio: "))
w_p_v = int(input("Enter weak pixel intensity value: "))
rmin = int(input("Enter minimum radius value: "))
rmax = int(input("Enter maximum radius value: "))
threshold = float(input("Enter threshold value for finding circles: "))

I = cv2.imread(name)

E_I = canny(I, size, sigma, l_t_r, h_t_r, w_p_v)

line_detection(name, I, E_I)

circle_detection(name, E_I, rmin, rmax, threshold)
