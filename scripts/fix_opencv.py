"""Make sure OpenCV (cv2) is installed correctly; repair it if not.

Colab ships with opencv-python, and MediaPipe pulls in opencv-contrib-python. Both install
into the same ``cv2`` folder, and when pip swaps one for the other the folder can be left
half empty: ``import cv2`` still works but ``cv2.CascadeClassifier`` or
``cv2.VideoCapture`` are missing. This script checks for that in a fresh process and, if
needed, removes every OpenCV package and installs a single clean copy.

    python scripts/fix_opencv.py
"""

import subprocess
import sys

CHECK = (
    "import cv2; "
    "[getattr(cv2, n) for n in ('VideoCapture', 'CascadeClassifier', 'FaceDetectorYN', 'imwrite')]; "
    "print('OpenCV', cv2.__version__, 'OK')"
)
VARIANTS = ["opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"]
CLEAN = "opencv-contrib-python"  # the variant MediaPipe depends on; includes everything we use


def ok() -> bool:
    return subprocess.run([sys.executable, "-c", CHECK]).returncode == 0


def main() -> int:
    if ok():
        return 0
    print("OpenCV is broken (two OpenCV packages overwrote each other). Reinstalling one clean copy...")
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(pip + ["uninstall", "-y", "-q", *VARIANTS], check=False)
    subprocess.run(pip + ["install", "-q", "--no-cache-dir", CLEAN], check=True)
    if ok():
        return 0
    print("OpenCV is still broken: use Runtime > Disconnect and delete runtime, then run Setup again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
