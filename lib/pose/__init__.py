import os
import platform
import sys

current_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if platform.system() == "Windows":
    os.environ["path"] += ";" + os.path.join(current_dir, "lib/models/pose")
elif platform.system() == "Linux":
    sys.path.append(os.path.join(current_dir, "lib/models/pose"))

from lib.pose.poselandmark import PoseLandmark
from lib.pose.poselandmarktrt import PoseLandmarkTRT
