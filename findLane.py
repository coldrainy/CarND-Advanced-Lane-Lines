import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# %matplotlib qt

##calibrate the camera
#draw corners
import glob
import numpy as np
img_names = glob.glob('camera_cal/calibration*.jpg')
objpoints = []
imgpoints = []

objpoint = np.zeros((6*9,3),np.float32)
objpoint[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

for img_name in img_names:
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(9,6),None)
    if ret==True:
        objpoints.append(objpoint)
        imgpoints.append(corners)
        image = cv2.drawChessboardCorners(img,(9,6),corners,ret)
        cv2.imshow('image',img)
        cv2.waitKey(100)
    else:
        print(img_name,ret)
cv2.destroyAllWindows()

#calibration
test_img = mpimg.imread('camera_cal/calibration1.jpg')
img_size = (test_img.shape[1],test_img.shape[0])
print(img_size)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(test_img,mtx,dist,None,mtx)

cv2.imshow('dst',dst)
cv2.waitKey(100)
cv2.destroyAllWindows()

def warp(img,points_src,points_dist):
    img_size = (img.shape[1],img.shape[0])
    #self lane
    src = np.float32(points_src)
    dist = np.float32(points_dist)
    #neibor lane
    #src = np.float32([[845,460],[1010,500],[770,500],[705,460]])
    #dist = np.float32([[700,300],[700,500],[500,500],[500,300]])
    """
    #another lane
    src = np.float32([[845,460],[1010,500],[517,500],[575,460]])
    dist = np.float32([[400,400],[400,600],[200,600],[200,400]])
    """
    M = cv2.getPerspectiveTransform(src, dist)
    Minv = cv2.getPerspectiveTransform(dist, src)
    #Minv = 0
    warped = cv2.warpPerspective(img,M,img_size)
    return warped,M,Minv

image = mpimg.imread('test_images/test1.jpg')
dst = cv2.undistort(image,mtx,dist,None,mtx)

points_src = [[700,460],[1110,720],[200,720],[580,460]]
points_dist = [[850,0],[850,720],[350,720],[350,0]]
warped,M,Minv = warp(dst,points_src,points_dist)

def abs_sobel_thres(img,orient='x',thresh_min=0,thresh_max=255):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobel = []
    if(orient == 'x'):
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0)
    else:
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
def hls_select(img, thresh=(100, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
#     sobelx = cv2.Sobel(s,cv2.CV_64F,1,0,ksize = 3)

#     abs_sobel = np.absolute(sobelx)
#     scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output[(s>thresh[0])&(s<=thresh[1])]=1
    # 3) Return a binary image of threshold result
    return binary_output
def h_select(img, thresh=(15, 100)):
    # 1) Convert to HSV color space
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # 2) Apply a threshold to the S channel
    v = hsv[:,:,2]
    binary_output = np.zeros_like(v)

    binary_output[(v>thresh[0])&(v<=thresh[1])]=1
    # 3) Return a binary image of threshold result
    return binary_output
def r_select(img,thresh=(0,255)):
    r = img[:,:,0]
    sobelx = cv2.Sobel(r,cv2.CV_64F,1,0,ksize = 3)

    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(r)

    binary_output[(r>thresh[0])&(r<=thresh[1])]=1
    return binary_output
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
def get_warped_binary(dst):
    abs_binary = mag_thresh(dst,sobel_kernel=3, mag_thresh=(30, 100))
    gradx_binary = abs_sobel_thres(dst, orient='x', thresh_min=20,thresh_max=100)
    grady_binary = abs_sobel_thres(dst, orient='y', thresh_min=40,thresh_max=100)
    h_binary = h_select(dst, thresh=(200, 255))
    s_binary = hls_select(dst, thresh=(100, 255))
    combined = np.zeros_like(abs_binary)
    #combined[((gradx_binary == 1)|(s_binary==1)) | ((abs_binary == 1) & (h_binary == 1))] = 1
    # combined[(((gradx_binary == 1)|(s_binary==1)) | ((abs_binary == 1) & (h_binary == 1)))& gradx_binary == 1] = 1
    combined[(((h_binary == 1)|(abs_binary == 1)) & ((s_binary==1) | (gradx_binary == 1)))] = 1

    binary_warped = cv2.warpPerspective(combined,M,(combined.shape[1],combined.shape[0]))
    return binary_warped
def get_current_x(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
#     out_img = np.dstack((binary_warped, binary_warped, binary_warped))*10
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    return leftx_current,rightx_current
def fit_laneline(binary_warped,leftx_current,rightx_current):

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 20
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
#         # Draw the windows on the visualization image
#         cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
#         (0,10,0), 2)
#         cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
#         (0,10,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit,right_fit
def draw_lane(binary_warped,left_fit,right_fit):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    lane_windows_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    lane_windows_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lane_pts = np.hstack((lane_windows_left, lane_windows_right))

    cv2.fillPoly(color_warp, np.int_([lane_pts]), (0,255, 0))

    lane = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1],color_warp.shape[0]), flags=cv2.INTER_LINEAR)
    return lane
def calculate_param(left_fit,right_fit,y_eval):
    # Define conversions in x and y from pixels space to meters

    ym_per_pix = 50/360 # meters per pixel in y dimension
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad,right_curverad
def process_image(img):

    binary_warped = get_warped_binary(img)
    if (l_line.detected == True) and (r_line.detected == True):
        leftx_current = l_line.bestx
        rightx_current = r_line.bestx
    else:
        leftx_current,rightx_current = get_current_x(binary_warped)
    left_fit,right_fit = fit_laneline(binary_warped,leftx_current,rightx_current)
    lane_img = draw_lane(binary_warped,left_fit,right_fit)
    result = cv2.addWeighted(img, 1, lane_img, 0.2, 0)
    xm_per_pix = 10/1280 # meters per pixel in x dimension
    left_curverad,right_curverad = calculate_param(left_fit,right_fit,img.shape[1])
    offset = ((rightx_current - leftx_current)-img.shape[1]//2)*xm_per_pix
    cv2.putText(result, 'Radius of Curvature = ' +  str(left_curverad) + 'm', (300, 45) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)
    cv2.putText(result, 'Vehicel is ' + str(offset) + 'm' + ' of center.',(300, 95) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)
    return result

from moviepy.editor import VideoFileClip
from IPython.display import HTML
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
l_line = Line()
r_line = Line()

video_output1 = 'project_video_output.mp4'
clip3 = VideoFileClip('project_video.mp4')
project_clip = clip3.fl_image(process_image)
project_clip.write_videofile(video_output1, audio=False)
