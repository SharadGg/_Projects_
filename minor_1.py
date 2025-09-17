from opencv_python import cv2


def open_webcam_with_fallback(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("Could not open webcam. Try a different index (1/2) or check permissions.")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed.")
                break

            try:
                cv2.imshow("Webcam (press Q to quit)", frame)
            except cv2.error as e:
                print("OpenCV display error:", e)
                print("Tip: If running headless/remote, GUI windows may not be supported.")
                break

            # Exit on 'q' or if the window is closed
            key = (cv2.waitKey(1) & 0xFF)
            if key in (ord('q'), ord('Q')):
                break
            if cv2.getWindowProperty("Webcam (press Q to quit)", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_webcam_with_fallback(0)