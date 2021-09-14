
import sys 
BASE_DIR = "/media/ders/zhangyumin/ritm_interactive_segmentation/"
sys.path.append(BASE_DIR)
# import models 
from  export import init_predictor
from  hRnet.predictor import Hr_predictor
from hRnet.baseclicker import Clicker,Click