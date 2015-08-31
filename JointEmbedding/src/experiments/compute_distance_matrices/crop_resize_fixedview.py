import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from global_variables import *

# parallely crop and resize images
os.system("find %s -name '*.png'|xargs -n 1 -P 40 mogrify -trim -resize '227*227!'" % (g_fixedview_image_folder))
