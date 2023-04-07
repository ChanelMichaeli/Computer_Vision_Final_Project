import numpy as np
import cv2
from scipy.signal import medfilt2d
import time
from matplotlib import pyplot as plt
from skimage.transform import resize


# Define a function called Canny that takes in an image as a parameter
def Canny(img):
    """ Canny function find the edges of the given image.

    :param img: Image in RGB format
    :return: Edges of the image in Gray Scale format.
    """
    # Convert the image from RGB format to Gray Scale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Convert the image to 8-bit unsigned integer
    img = img.astype(np.uint8)
    # Detect edges in the image using the Canny algorithm with low and high thresholds of 100 and 150
    edges = cv2.Canny(img, 100, 150)
    # Return the detected edges
    return edges


# Define a function called fill_contours that takes in a NumPy array as a parameter
def fill_contours(arr):
    """ This function fill the given contour (most of the time the edges of image from Canny function).

       :param arr: Contour as an array (one of the contour from the edges of the image).
       :return: fill the contour.
       """
    # Apply maximum accumulation along axis 1 of the array
    max_acc_arr = np.maximum.accumulate(arr, 1)
    # Reverse the array along axis 1 and apply maximum accumulation along axis 1
    max_acc_arr_rev = np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1]
    # Compute the element-wise bitwise AND of the two arrays
    filled_arr = max_acc_arr & max_acc_arr_rev
    # Return the filled array
    return filled_arr


def ball_area(img, factor, flag, max_radius):
    """ ball area function find the ball area form the given image. the function find the ball
            by using Hough Circle transform of by detecting the Orange/Yellow color of the ball.
            After the function detect the location of the ball the function return the crop image
            that contain only the ball and the small area around it.

        :param :img: The given image
        :param factor: Determined the size of the crop image around the ball (small factor for small area of the ball
                       and bigger factor for bigger area of the ball)
        :param flag: Getting 0 or 1. if the flag = 0 its mean that 'find_circles' function find the circle
                    of the ball by Hough Circle transform and crop the image by it coordinates,
                    else flag = 1 and the image will be cropped by color estimated coordinate of the ball.
                    (The coordinate of the ball is [x_center, y_center, r_ball])
        :param max_radius: the threshold maximum radius in using the 'find_circles' function.
        :return: crop image around the ball.
        """
    if flag == 1:
        factor += 1
        _, x_center, y_center, r_ball = find_orange(img)
        if [x_center, y_center, r_ball] == [0, 0, 0]:
            crop_img = None  # if the 'find_orange' function didn't find any orange color in the image
            return crop_img, None, None
    else:
        _, x_center, y_center, r_ball = Circles(img, max_radius)
    height, width = img.shape[:2]  # Find the 'height' and 'width' of the image

    # Determine crop bounds
    x_min = max(0, x_center - r_ball * factor)
    x_max = min(width, x_center + r_ball * factor)
    y_min = max(0, y_center - r_ball * factor)
    y_max = min(height, y_center + r_ball * factor)

    # Crop the image
    cropped_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

    # Return the cropped image and the coordinates of the ball.
    return cropped_img, x_center, y_center


def find_circles(img, channel, minRadius, maxRadius):
    """ This function find the circles of the ball or the balloon, depend on the min/max Radius threshold
            parameters.

        :param img: The given image
        :param channel: The channel of the HSV image that the function get after converting the image to HSV
        :param minRadius: The minimum radius threshold
        :param maxRadius: The maximum radius threshold
        :return: All the circles in the threshold range, The radius of all the circle, The channel of the HSV image
        """
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract the specified channel from the HSV image
    hsv_channel = hsv_img[:, :, channel]

    # Convert the channel to a 3-channel grayscale image
    cimg = cv2.cvtColor(hsv_channel, cv2.COLOR_GRAY2BGR)

    # Use the HoughCircles function to detect circles in the grayscale image
    circles = cv2.HoughCircles(hsv_channel, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=minRadius,
                               maxRadius=maxRadius)

    # Return the detected circles, the radii array, and the grayscale channel image
    return circles, np.zeros(img.shape), hsv_channel


def Circles(img, max_radius):
    """ This function get image and return the contour of the circle, and it coordinates
            (The coordinates are: [x, y, radius])
        :param img: The given image
        :param max_radius: the threshold maximum radius in using the 'find_circles' function.
        :return: the circles, the center coordinate of the ball (x, y) and the radius of the circles
        """
    # Find circles in the image using HoughCircles algorithm
    circles = cv2.HoughCircles(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2], cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                               param2=30, minRadius=0, maxRadius=max_radius)

    # If no circle is found, return None
    if circles is None:
        print('Miss')
        return None, None, None, None

    # Convert the coordinates and radius of the circle to integers
    circles = np.uint16(np.around(circles))

    # Create a blank image of the same size as the input image
    rad = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Draw the detected circle(s) on the blank image
    for i in circles[0, :]:
        rad = cv2.circle(rad, (i[0], i[1]), i[2], (255, 255, 255), 2)

    # Convert the resulting image to grayscale
    rad_gray = cv2.cvtColor(rad, cv2.COLOR_RGB2GRAY)

    # Extract the center and radius of the first detected circle
    x = circles[0][0][0]
    y = circles[0][0][1]
    r = circles[0][0][2]

    # Return the resulting grayscale image, center x and y coordinates, and radius
    return rad_gray, x, y, r


# Define a function called ball_mask that takes in four parameters
def ball_mask(img, add_to_radius, flag, max_radius):
    """ This function getting the coordinate of the orange ball and the radius by Hough Circle
        or by color (depended on the flag) and create circle mask that have the same center coordinate
        and the same radius plus adding the parameter 'add_to_radius'

    :param img: The given image
    :param add_to_radius: adding to the original radius.
    :param flag: flag value is 0 or 1.
    :param max_radius: the threshold maximum radius in using the 'find_circles' function.
    :return: ball mask radius that bigger then the ball we found (if add_to_radius in NOT 0)
    """
    # Check if the flag is 1, then find the center and radius of the ball in the image
    if flag == 1:
        _, x_center, y_center, r_ball = find_orange(img)
        center = (x_center, y_center)
        radius = r_ball
    # If the flag is not 1, then find the circles in the image with a maximum radius
    else:
        ball_img, a, b, r = Circles(img, max_radius)
        center = (a, b)
        # Add the given value to the radius
        radius = r + add_to_radius

    # Create a matrix of zeros with the same dimensions as the input image
    matrix = np.zeros((img.shape[0], img.shape[1]))
    # Create a meshgrid of x and y values for each pixel in the matrix
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    # Create a mask that marks the pixels inside the ball region as 1
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2).astype(int)
    # Multiply the mask with a matrix of ones to get a new ball image with only the ball pixels marked as 1
    new_ball = np.multiply(np.ones_like(matrix), mask)
    # Return the new ball image
    return new_ball


def miss_or_maybe(img):
    """ This function check if the balloon is close to the ball or not.
        If the balloon us close to the ball the function return crop image around the ball so the crop
        image will contain the balloon and the ball.

    :param img: The given image
    :return: the crop image that contain the ball and the balloon or a black picture.
    """
    img = only_screen(img, 25, 160, 255)
    max_radius = 20

    # Use a loop to avoid repetitive code
    for i in range(10):
        circles, _, _ = find_circles(img, 2, 0, max_radius)
        if circles is not None:
            break
        max_radius += 1

    if circles is None:
        flag = 1
    else:
        flag = 0

    # Use the flag variable to avoid repetitive code
    crop_img, _, _ = ball_area(img, 1 if flag == 1 else 2.5, flag, max_radius)
    if crop_img is None:
        img_around_ball = np.zeros((img.shape[0], img.shape[1]))
        return img_around_ball, img_around_ball, max_radius, flag
    edge_img = Canny(crop_img)
    # Use the flag variable to avoid repetitive code
    if flag == 1:
        edge_img[edge_img == 255] = 1
        mask = ball_mask(crop_img, 20, 1, max_radius)
    else:
        height, width = edge_img.shape[:2]
        mask = full_contour(edge_img)
        # Find the indices of the non-zero elements
        row_indices, col_indices = np.nonzero(mask)

        # Convert the indices to a list of tuples
        mask_nozero = [(row_indices[i], col_indices[i]) for i in range(len(row_indices))]

        x_max_mask = max(mask_nozero, key=lambda x: x[1])[1]
        x_min_mask = min(mask_nozero, key=lambda x: x[1])[1]
        y_max_mask = max(mask_nozero, key=lambda x: x[0])[0]
        y_min_mask = min(mask_nozero, key=lambda x: x[0])[0]
        if x_max_mask == width - 1 or x_min_mask == 0 or y_max_mask == height - 1 or y_min_mask == 0:
            mask = ball_mask(crop_img, 12, 0, max_radius)
        if mask is None:
            mask = ball_mask(crop_img, 20, 0, max_radius)
    rectangle_mask = np.zeros((crop_img.shape[0], crop_img.shape[1]))
    rectangle_mask[3:-3, 3:-3] = 1
    no_ball = edge_img * np.logical_not(rectangle_mask)
    if np.max(no_ball) == 0:
        img_around_ball = np.zeros((img.shape[0], img.shape[1]))
        return img_around_ball, img_around_ball, max_radius, flag
    img_around_ball, _, _ = ball_area(img, 7, flag, max_radius)
    edge_cropped_img = Canny(img_around_ball)
    full_balloon = full_contour(edge_cropped_img)
    s = np.sum(full_balloon == 1)
    print('s is:', s)
    if s < 2000:
        img_around_ball = np.zeros((img.shape[0], img.shape[1]))
        return img_around_ball, img_around_ball, max_radius, flag
    if np.sum(full_balloon == 1) - np.sum(mask == 1) == 0:
        img_around_ball = np.zeros((img.shape[0], img.shape[1]))
        return img_around_ball, img_around_ball, max_radius, flag
    else:
        return img_around_ball, full_balloon, max_radius, flag


def full_contour(img):
    """ This function return the biggest contur of the given image and fill it.

    :param img: The given image
    :return: The biggest full contour
    """
    # Find all contours in the edges image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables
    max_size = 0
    contour_balloon = None

    # Iterate through all contours and find the one with the maximum size
    for contour in contours:
        contour_image = cv2.drawContours(np.zeros_like(img), [contour], 0, 1, 5)
        contour_image = fill_contours(contour_image)
        size = np.count_nonzero(contour_image)
        if size > max_size:
            max_size = size
            contour_balloon = contour_image

    # Return the binary image containing the full contour of the object
    return contour_balloon


def only_screen(img, factor, low_range, high_range):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a binary threshold to the grayscale image
    _, binary_img = cv2.threshold(gray_img, low_range, high_range, cv2.THRESH_BINARY)

    # Apply morphological opening to remove small objects and smooth the edges
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Find contours in the opening image
    contours, _ = cv2.findContours(opening_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = x + factor
    w = w - 1.5*factor
    # Crop the image using the bounding rectangle
    crop_img = img[y:y + int(h), x:x + int(w)]
    return crop_img


def find_orange(img):
    """ This function detect the orange ball by color of orange range and yellow range.

    :param img: The given image
    :return: binary image of the orange ball
    """
    # Convert image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define lower and upper bounds for yellow and orange color
    yellow_lower = np.array([0, 50, 50])
    yellow_upper = np.array([40, 255, 255])
    orange_lower = np.array([0, 50, 50])
    orange_upper = np.array([35, 255, 255])

    # Create mask for yellow color and apply to image
    yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    yellow_color = cv2.bitwise_and(img, img, mask=yellow_mask)
    _, yellow_color = cv2.threshold(yellow_color, 0, 255, cv2.THRESH_BINARY)

    # Create mask for orange color and apply to image
    mask = cv2.inRange(hsv_img, orange_lower, orange_upper)
    orange_color = cv2.bitwise_and(img, img, mask=mask)
    _, orange_color = cv2.threshold(orange_color, 0, 255, cv2.THRESH_BINARY)

    # Combine yellow and orange masks to get ball mask
    ball = yellow_color + orange_color

    # Apply median filter to each color channel of the ball mask
    ball[:, :, 0] = medfilt2d(1 * ball[:, :, 0], kernel_size=9)
    ball[:, :, 1] = medfilt2d(1 * ball[:, :, 1], kernel_size=9)
    ball[:, :, 2] = medfilt2d(1 * ball[:, :, 2], kernel_size=9)

    # If there are no non-zero pixels in the ball mask, return empty image
    if np.max(ball) == 0:
        img_around_ball = np.zeros((img.shape[0], img.shape[1]))
        return img_around_ball, 0, 0, 0

    # Find bounding box of ball mask and compute center and estimated radius
    no_zero = np.nonzero(ball)
    x_max = np.max(no_zero[1])
    x_min = np.min(no_zero[1])
    y_max = np.max(no_zero[0])
    y_min = np.min(no_zero[0])
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    estimate_r = max(delta_x, delta_y) / 2

    return ball, int(x_center), int(y_center), int(estimate_r)


def hit_miss(path):
    # Read and convert the image to RGB color space
    original_img_color = cv2.imread(path, -1)
    original_img_color = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2RGB)

    # Adjust the brightness of the image
    original_img_color = cv2.add(original_img_color, -50)
    # Get the zoomed image, full balloon mask and max radius
    zoom, full_balloon, max_radius, flag = miss_or_maybe(original_img_color)

    # Check if there is a balloon in the image
    if np.max(zoom) == 0:
        print('Miss')
        return False

    # Find the orange ball using HoughCircles or ball_mask function
    circles, _, _ = find_circles(zoom, 2, 0, max_radius)
    if circles is None:
        orange_ball = ball_mask(zoom, 2, 1, max_radius)
    else:
        orange_ball, _, _, _ = Circles(zoom, max_radius)
        orange_ball = fill_contours(orange_ball)

    # Convert the orange ball mask to binary (1 for orange pixels and 0 for non-orange pixels)
    orange_ball[orange_ball == 255] = 1

    # Compute the overlap between the orange ball and the full balloon
    overlap = orange_ball * full_balloon
    overlap[overlap > 255] = 255

    # Compute the number of pixels in the ball
    num_pixels_ball = np.sum(orange_ball == 1)

    # Check if the overlap pixels > 10% of the pixels in the ball
    if np.sum(overlap == 1) >= num_pixels_ball / 10:
        print("Pop the balloon!")
        return True
    else:
        print("Miss")
        return False


start_time = time.time()
hit_miss('/Users/NoanAtias/PycharmProjects/FinalProject/error4.jpeg')
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")