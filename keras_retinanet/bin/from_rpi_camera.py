import cv2


def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def main():
    stream_string = gstreamer_pipeline(flip_method=0)
    print(stream_string)

    cap = cv2.VideoCapture(stream_string)

    if not cap.isOpened():
        raise Exception("Unable to open camera")

    window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

    while cv2.getWindowProperty('CSI Camera',0) >= 0:
        ret_val, img = cap.read();
        cv2.imshow('CSI Camera',img)
        keyCode = cv2.waitKey(30) & 0xff

        # Stop the program on the ESC key
        if keyCode == 27:
            break

        sleep(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()