import cv2



# function to display the coordinates of
# of the points clicked on the image

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks

    laser_line_color = (0, 0, 255)

    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        cv2.circle(img,(x,y),5,(0,255,255),thickness=-1)
        n=len(X)
        if n >= 1:
            cv2.line(img, (x, y), (X[n-1], Y[n-1]), laser_line_color, 2)
        X.append(x)
        Y.append(y)

        cv2.imshow('image', img)


# driver function
if __name__ == "__main__":
    imH = 500
    imW = 700
    # reading the image
    cap = cv2.VideoCapture("image-test/cam2.mp4")

    # Doc frame dau tien de nguoi dung chon doi truong can track
    ret, img = cap.read()
    #img = cv2.imread('lena.jpg', 1)
    img = cv2.resize(img, (imW, imH))
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse hadler for the image
    # and calling the click_event()
    X = []
    Y = []
    cv2.setMouseCallback('image', click_event)
    n=len(X)


    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()