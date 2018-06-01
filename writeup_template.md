**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 9 through 30 of the file called  _findLane.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoint` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![calibration](/output__image/undistort_output.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![calibration](/output__image/1.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.I defined the threshold function at lines 69 through 147 in findLane.py.And use them to get warped binary image in the function get_warped_binary at line 171 through 183. Here's an example of my output for this step.

![threshold_binary](/output__image/threshold_binary.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 43 through 60 in the file `fineLane.py`.  The `warp()` function takes as inputs an image (`img`), as well as source (`points_src`) and destination (`points_dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
points_src = np.float32([[700,460],[1110,720],[200,720],[580,460]])
points_dist = np.float32([[850,0],[850,720],[350,720],[350,0]])
```


I verified that my perspective transform was working as expected by drawing the `points_src` and `points_dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![perspective][/output__image/perspective.png)]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I implement the sliding window method to find the lane-line pixels and fit my lane lines with a 2nd order polynomial(at the line 199 throuth 258 in the findLane.py file) kinda like this:

![lane_line_pixels][/output__image/lane_line_pixels.png)]
![polynomial][/output__image/polynomial.png)]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius in lines 274 through 280 in my code in `fineLane.py` and the position of the vehicle with respect to the lane line center in the line 294 of the fineLane.py center in the line 294 of the fineLane.py.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I plotted the result back down onto the road in lines 259 through 273 in my code in `fineLane.py` in the function `draw_lane(binary_warped,left_fit,right_fit)`.  Here is an example of my result on a test image:

![lane][/output__image/lane.png)]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Well,my pipeline worded well in the project_video.But it can't get a good result in the other two videos.It is still sensitive to the light.And one thing I could do to make my pipeline more robust is to use the history information of the line.I should do this but there isn't enough time
to accomplish it.I think i will work on it after finishing this term.
