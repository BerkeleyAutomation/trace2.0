import sys
sys.path.append('/home/justinyu/multicable-decluttering/yumi_jacobi')
sys.path.append('/home/justinyu/multicable-decluttering/')

# from yumi_jacobi.interface import Interface
from utils.scripts.brio.brio_sensor import BRIOSensor

YUMI_DIR = '/home/justinyu/multicable-decluttering/yumi_jacobi/starter_examples/AUTOLAB_BWW_YuMi.jacobi-project'
SAVE_DIR = '/home/justinyu/multicable-decluttering/decluttering/src/ICRA_FIGURES/'
YUMI_MIN_POS = [-2.94, -2.350, -2.94, -2.16, -5.00, -1.54, -3.99]
YUMI_MAX_POS = [2.9409, 0.7592, 2.9409, 1.2, 5.00, 2.4086, 3.8]

#pendant to jacobi: [1:1, 2:2, 7:3, 3:4, 4:5, 5:6, 6:7]

# IP Vectors = VECTOR_SCALE * (divergence_pt - start_pt)
VECTOR_SCALE = 2.0
# Decrease the KEEPOUT_DIST to push IP Vectors in closer
KEEPOUT_DIST = 12 #10

# Threshold above which trace is considered part of knot
### Original: 0.75
DENSITY_THRESH = 0.75

# Radius in which to calculate density along every trace
DENSITY_RADIUS = 100

NUM_MOVES = 10
CC_RADIUS = 100
MIN_REGION_AREA = 80

RED     = (255, 0, 0)
GREEN   = (0, 255, 0)
BLUE    = (0, 0, 255)
MAGENTA = (234, 52, 175)
YELLOW  = (255, 234, 52)
CYAN    = (52, 175, 234)
ORANGE  = (243, 123, 0)
BROWN   = (132, 64, 32)
WHITE   = (255, 255, 255)
PURPLE  = (148, 0, 211)
PINK    = (255, 105, 180)
TEAL    = (0, 128, 128)
LIME    = (50, 205, 50)
CORAL   = (255, 127, 80)
GOLD    = (255, 215, 0)
INDIGO  = (75, 0, 130)
SILVER  = (192, 192, 192)
COLORS  = [RED, GREEN, BLUE, MAGENTA, YELLOW, CYAN, ORANGE, BROWN,
           PURPLE, PINK, TEAL, LIME, CORAL, GOLD, INDIGO, SILVER]