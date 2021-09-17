import cv2
import sys
import glob
import webcolors
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from collections import Counter

############Color Identification############

#RGB to Hex Conversion
def rgb2hex(c):
    return "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))  # format(int(c[0]), int(c[1]), int(c[2]))

#Hex code to name conversion
def hex2name(c):
    h_color = '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))
    try:
        nm = webcolors.hex_to_name(h_color, spec='css3')
    except ValueError as v_error:
        print("{}".format(v_error))
        rms_lst = []
        for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
            cur_clr = webcolors.hex_to_rgb(img_hex)
            rmse = np.sqrt(mean_squared_error(c, cur_clr))
            rms_lst.append(rmse)

        closest_color = rms_lst.index(min(rms_lst))
        nm = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]
    return nm

#Read image in RGB color space
def get_image(image_path):
    image = cv2.imread(image_path)
    cv2.imshow('Sample Image', image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

#Get colors from an image
def get_colors(image, no_of_colors, show_chart):
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
    color = KMeans(n_clusters = no_of_colors)
    label = color.fit_predict(modified_image)
    cnt = Counter(label)
    center_color = color.cluster_centers_

# We get ordered colors by iterating through the keys
    ord_color = [center_color[i] for i in cnt.keys()]
    hex_color = [rgb2hex(ord_color[i]) for i in cnt.keys()]
    lbl_color = [hex2name(ord_color[i]) for i in cnt.keys()]

    if(show_chart):
        plt.pie(cnt.values(), labels=lbl_color, colors=hex_color)
        plt.show()

get_colors(get_image('images\\shapes.png'), 5, True)
get_colors(get_image('images\\flower.jpg'), 8, True)