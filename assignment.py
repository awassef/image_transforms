import cv2
import numpy as np
import matplotlib.pyplot as plt


def negative(img):
    negative_img = -img + 255
    input_values = list(range(0, 256))
    output_values = list(range(255, -1, -1))
    mapping = dict(zip(input_values, output_values))
    return negative_img, mapping

def power_transform(img, c=1, gamma=0.6):
    img = img / 255
    img = c * (img**gamma)
    img = img * 255
    img = img.astype(np.uint8)

    input_list = list(range(0, 256))
    output_list = np.array(range(0, 256))
    output_list = (output_list / 255) ** gamma * c * 255
    mapping = dict(zip(input_list, list(output_list)))

    return img, mapping

def contrast_stretch(img):
    input_list = list(range(img.min(), img.max()))
    output_list = []
    for input_ in input_list:
        output_list.append(((input_ - img.min()) / (img.max() - img.min())) * 255)

    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255
    img = img.astype(np.uint8)
    mapping = dict(zip(input_list, output_list))
    return img, mapping

def grey_slice(img):

    img[(img > 50) & (img < 100)] = 200

    input_vals = list(range(0, 256))
    output_vals = np.array(range(0, 256))

    mask = output_vals > 50
    mask = mask & (output_vals < 100)
    output_vals[mask] = 200

    output_vals = list(output_vals)
    mapping = dict(zip(input_vals, output_vals))
    return img, mapping

def hist_equalize(img):

    max_val = int(img.max())
    intensity_list = []
    for intensity in range(img.min(), img.max()):
        n_intensity = sum(sum(img == intensity))
        intensity_list.append(n_intensity)

    sum_ = sum(intensity_list)

    curr_sum = 0

    mapping = {}
    curr_intensity = -1
    for n_intensity in intensity_list:
        curr_intensity += 1
        curr_sum += n_intensity 
        probability = curr_sum / sum_
        new_val = int(round(probability * max_val))
        mapping[curr_intensity] = new_val

    return mapping

def apply_mapping(img, mapping):
    img_copy = img
    for intensity, new_intensity in mapping.items():
        img_copy[img == intensity] = new_intensity
    return img_copy

def display_mapping(mapping, title):
    plt.plot(list(mapping.keys()), list(mapping.values()))
    plt.xlabel('input value, r')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.ylabel('output value, s')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('Resources/x-ray_angiogram.jpg', -1)
    negative_img, negative_mapping = negative(img)
    power_img, power_mapping = power_transform(img)
    stretched_img, stretched_mapping = contrast_stretch(img)
    sliced_img, sliced_mapping = grey_slice(img)

    cv2.imshow('negative', negative_img)
    cv2.imshow('power', power_img)
    cv2.imshow('stretched', stretched_img)
    cv2.imshow('sliced', sliced_img)

    img2 = cv2.imread('Resources/chest_x-ray1.jpg', -1)

    equalize_mapping = hist_equalize(img2)
    equalized = apply_mapping(img2, equalize_mapping) 

    cv2.imwrite('equalized.png', equalized)
    cv2.waitKey()

    display_mapping(negative_mapping, 'Negative Transform')
    display_mapping(sliced_mapping, 'Grey Level Slice Transform')
    display_mapping(power_mapping, 'Power Transform')
    display_mapping(stretched_mapping, 'Contrast Stretch Transform')
    display_mapping(equalize_mapping, 'Histogram Equalization Transform')

