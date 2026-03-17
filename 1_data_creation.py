import os
import cv2


def collect_images(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)

    counter = 0
    while True:

        ret, frame = cam.read()

        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"Captured: {counter} | SPACE=save  ESC=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Calibration Images", display)

        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            save_path = os.path.join(save_dir + f"/frame_{counter:04d}.png")
            counter += 1
            print(f"Saving image to path: {save_path}")

            cv2.imwrite(save_path, frame)

        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    output_dir = "output/calib/images"

    collect_images(output_dir)
