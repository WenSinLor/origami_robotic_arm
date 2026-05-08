"""
Robust Red Marker Tracker
=========================
Edit the CONFIG block below, then run:
    python tracker_fixed.py

Hotkeys during live preview:
    Q  —  quit
    S  —  save screenshot
    T  —  HSV tuner
"""

import cv2
import numpy as np
import sys
import h5py
from pathlib import Path
from collections import deque
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


# ═══════════════════════════════════════════════════════════════
#  CONFIG  —  edit this block only
# ═══════════════════════════════════════════════════════════════

BASE         = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"
VIDEO_FPS    = 30000.0 / 1001.0
SHOW_PREVIEW = True
QUAD_W       = 1280
QUAD_H       = 720

NUM_MARKERS  = 19
MIN_AREA     = 80
MAX_AREA     = 8000
MAX_DIST     = 180
MAX_LOST     = 15
BLUR_KERNEL  = 5
MORPH_SIZE   = 7
SHOW_IDS     = True

RED_LOWER_1  = (0,   75,  80)
RED_UPPER_1  = (10,  255, 255)
RED_LOWER_2  = (165, 75,  80)
RED_UPPER_2  = (180, 255, 255)

# One dict per angle. Keys are "M0".."M18", values are (x, y) pixel positions.
# HOW TO UPDATE: run pick_reference_frame.py on a still frame from that angle,
# press S, paste the printed dict here.
# HOW TO FIX A SWAP: just rename the key (e.g. "M7" <-> "M8").
# Coordinates stay as-is. Key order does not matter.

"""
Soft State 100g
"""
# REFERENCE_LAYOUTS = {

#     "coor_0": {
#         'M0': (1187.4, 88.9),
#         'M1': (765.2, 98.1),
#         'M2': (1022.7, 108.7),
#         'M3': (1103.1, 152.7),
#         'M4': (936.2, 159.6),
#         'M5': (831.7, 162.8),
#         'M6': (1041.8, 254.4),
#         'M7': (1178.1, 260.9),
#         'M8': (777.9, 266.8),
#         'M9': (943.9, 321.0),
#         'M10': (838.8, 347.3),
#         'M11': (1094.9, 359.2),
#         'M12': (773.7, 435.0),
#         'M13': (921.8, 446.2),
#         'M14': (1175.0, 451.5),
#         'M15': (1033.6, 491.5),
#         'M16': (833.9, 532.6),
#         'M17': (1112.7, 539.8),
#         'M18': (652.7, 809.9),
#     },

#     "coor_1": {
#         'M0': (1190.4, 83.7),
#         'M1': (763.5, 94.0),
#         'M2': (1023.6, 111.3),
#         'M3': (1109.0, 141.9),
#         'M5': (828.4, 152.8),
#         'M4': (936.4, 166.0),
#         'M7': (1183.0, 255.2),
#         'M8': (776.2, 263.8),
#         'M6': (1045.5, 264.2),
#         'M10': (835.7, 328.9),
#         'M11': (1102.0, 338.0),
#         'M9': (945.0, 338.3),
#         'M12': (771.8, 434.0),
#         'M14': (1181.9, 443.4),
#         'M13': (923.3, 475.0),
#         'M16': (831.5, 508.9),
#         'M17': (1123.4, 512.1),
#         'M15': (1038.0, 522.0),
#         'M18': (664.3, 624.7),
#     },

#     "coor_2": {
#         'M0': (1194.1, 72.3),
#         'M1': (767.6, 99.0),
#         'M2': (1025.0, 110.5),
#         'M3': (1114.1, 130.7),
#         'M4': (940.9, 168.9),   # Swapped with M5
#         'M5': (831.5, 154.5),   # Swapped with M4
#         'M6': (1055.7, 261.0),  # Swapped with M7
#         'M7': (1194.0, 232.7),  # Swapped with M6
#         'M8': (790.4, 277.9),
#         'M9': (962.5, 343.1),   # Swapped with M11
#         'M10': (850.2, 334.1),
#         'M11': (1118.1, 314.8), # Swapped with M9
#         'M12': (804.6, 458.2),  # Swapped with M13
#         'M13': (958.2, 480.6),  # Swapped with M15
#         'M14': (1211.8, 404.2), # Swapped with M12
#         'M15': (1079.3, 506.7), # Swapped with M16
#         'M16': (873.8, 524.4),  # Swapped with M17
#         'M17': (1162.8, 476.3), # Swapped with M14
#         'M18': (756.2, 803.1),
#     },

#     "coor_3": {
#         'M0': (1192.2, 79.0),
#         'M1': (770.0, 104.0),
#         'M2': (1024.9, 106.8),
#         'M3': (1109.8, 144.7),
#         'M4': (940.4, 161.4),
#         'M5': (838.1, 166.6),
#         'M6': (1052.4, 249.6),  # Swapped with M7
#         'M7': (1190.3, 240.6),  # Swapped with M6
#         'M8': (793.6, 282.0),
#         'M9': (960.4, 326.3),
#         'M10': (857.6, 355.6),  # Swapped with M11
#         'M11': (1115.4, 339.4), # Swapped with M10
#         'M12': (809.0, 461.7),  # Swapped with M14
#         'M13': (955.5, 457.1),
#         'M14': (1206.9, 415.0), # Swapped with M12
#         'M15': (1072.7, 482.4),
#         'M16': (881.2, 548.4),  # Swapped with M17
#         'M17': (1158.5, 505.7), # Swapped with M16
#         'M18': (830.2, 969.2),
#     },

# }

"""
Soft State 40g
"""
# REFERENCE_LAYOUTS = {

#     "coor_0": {
#         'M0': (1178.2, 81.6),
#         'M1': (751.1, 96.2),
#         'M2': (1010.1, 106.5),
#         'M3': (1095.9, 144.7),
#         'M4': (923.5, 158.6),
#         'M5': (818.4, 159.1),
#         'M6': (1032.9, 251.1),  # Swapped with M7
#         'M7': (1171.0, 248.2),  # Swapped with M6
#         'M8': (767.1, 266.5),
#         'M9': (935.0, 322.1),
#         'M10': (829.4, 340.3),
#         'M11': (1092.0, 342.5),
#         'M12': (768.9, 436.4),  # Swapped with M13
#         'M13': (918.1, 450.7),  # Swapped with M14
#         'M14': (1175.0, 431.3), # Swapped with M12
#         'M15': (1033.2, 490.0),
#         'M16': (833.0, 524.2),  # Swapped with M17
#         'M17': (1119.0, 515.0), # Swapped with M16
#         'M18': (680.3, 809.3),  
#     },

#     "coor_1": {
#         'M0': (1177.0, 82.4),
#         'M1': (750.2, 96.3),
#         'M2': (1009.1, 110.8),
#         'M3': (1096.1, 141.6),
#         'M4': (922.8, 164.2),   # Swapped with M5
#         'M5': (815.5, 155.0),   # Swapped with M4
#         'M6': (1031.9, 258.7),  # Swapped with M7
#         'M7': (1169.6, 248.5),  # Swapped with M6
#         'M8': (765.5, 266.0),
#         'M9': (933.7, 332.2),   # Swapped with M10
#         'M10': (825.4, 330.7),  # Swapped with M9
#         'M11': (1091.5, 333.1),
#         'M12': (766.4, 435.5),  # Swapped with M13
#         'M13': (917.3, 465.4),  # Swapped with M14
#         'M14': (1173.4, 430.4), # Swapped with M12
#         'M15': (1032.0, 505.5), # Swapped with M16
#         'M16': (828.2, 511.4),  # Swapped with M17
#         'M17': (1118.4, 503.3), # Swapped with M15
#         'M18': (667.3, 735.2), 
#     },

#     "coor_2": {
#         'M0': (1181.2, 79.4),
#         'M1': (755.6, 101.5),
#         'M2': (1012.5, 111.9),
#         'M3': (1099.7, 139.6),
#         'M4': (927.5, 166.7),   # Swapped with M5
#         'M5': (821.2, 158.9),   # Swapped with M4
#         'M6': (1039.1, 258.0),  # Swapped with M7
#         'M7': (1177.4, 239.9),  # Swapped with M6
#         'M8': (776.0, 275.4),
#         'M9': (944.8, 335.8),   # Swapped with M10
#         'M10': (837.5, 337.5),  # Swapped with M11
#         'M11': (1102.0, 325.4), # Swapped with M9
#         'M12': (786.2, 450.8),  # Swapped with M13
#         'M13': (937.3, 471.3),  # Swapped with M14
#         'M14': (1190.9, 414.1), # Swapped with M12
#         'M15': (1054.7, 501.5), # Swapped with M16
#         'M16': (853.6, 521.4),  # Swapped with M17
#         'M17': (1140.6, 488.6), # Swapped with M15
#         'M18': (720.2, 788.0),  
#     },

#     "coor_3": {
#         'M0': (1180.1, 78.1),
#         'M1': (754.8, 100.9),
#         'M2': (1011.2, 107.1),
#         'M3': (1098.9, 142.0),
#         'M4': (926.0, 160.7),
#         'M5': (822.3, 161.5),
#         'M6': (1037.8, 250.0),  # Swapped with M7
#         'M7': (1176.8, 239.7),  # Swapped with M6
#         'M8': (775.9, 275.2),
#         'M9': (943.4, 325.7),
#         'M10': (840.7, 344.1),  # Swapped with M11
#         'M11': (1102.3, 333.5), # Swapped with M10
#         'M12': (787.8, 450.9),  # Swapped with M13
#         'M13': (935.2, 456.9),  # Swapped with M14
#         'M14': (1190.1, 415.1), # Swapped with M12
#         'M15': (1052.3, 486.4), 
#         'M16': (857.0, 532.4),  # Swapped with M17
#         'M17': (1141.1, 499.9), # Swapped with M16
#         'M18': (760.0, 898.1),
#     },

# }

"""
Soft State 20g
"""
# REFERENCE_LAYOUTS = {

#     "coor_0": {
#         'M0': (1180.0, 80.5),   
#         'M1': (752.5, 95.8),    
#         'M2': (1010.5, 107.2),  
#         'M3': (1099.1, 142.9),  
#         'M4': (924.0, 159.5),   # Swapped with M5
#         'M5': (819.1, 157.1),   # Swapped with M4
#         'M6': (1034.1, 251.4),  # Swapped with M7
#         'M7': (1173.1, 245.2),  # Swapped with M6
#         'M8': (769.5, 266.5),   
#         'M9': (936.5, 324.0),   
#         'M10': (831.6, 336.9),  # Swapped with M11
#         'M11': (1096.5, 336.1), # Swapped with M10
#         'M12': (772.9, 437.7),  # Swapped with M13
#         'M13': (921.8, 454.4),  # Swapped with M14
#         'M14': (1179.2, 425.0), # Swapped with M12
#         'M15': (1037.4, 491.0), 
#         'M16': (838.1, 520.9),  # Swapped with M17
#         'M17': (1125.7, 506.3), # Swapped with M16
#         'M18': (698.4, 821.8),  
#     },

#     "coor_1": {
#         'M0': (1179.4, 79.9),   
#         'M1': (752.0, 96.0),    
#         'M2': (1011.0, 108.9),  
#         'M3': (1098.2, 140.4),  
#         'M4': (924.7, 161.9),   # Swapped with M5
#         'M5': (817.9, 155.3),   # Swapped with M4
#         'M6': (1034.8, 254.8),  # Swapped with M7
#         'M7': (1172.9, 244.2),  # Swapped with M6
#         'M8': (769.2, 266.0),   
#         'M9': (937.4, 328.6),   
#         'M10': (829.2, 332.1),  # Swapped with M11
#         'M11': (1095.3, 330.8), # Swapped with M10
#         'M12': (772.2, 436.8),  # Swapped with M13
#         'M13': (923.2, 461.3),  # Swapped with M14
#         'M14': (1178.8, 423.7), # Swapped with M12
#         'M15': (1038.5, 498.6), 
#         'M16': (835.3, 513.7),  # Swapped with M17
#         'M17': (1125.0, 499.1), # Swapped with M16
#         'M18': (684.0, 769.9),  
#     },

#     "coor_2": {
#         'M0': (1180.9, 77.4),   
#         'M1': (753.9, 98.4),    
#         'M2': (1011.6, 108.7),  
#         'M3': (1099.7, 138.4),  
#         'M4': (926.2, 162.8),   # Swapped with M5
#         'M5': (819.6, 156.9),   # Swapped with M4
#         'M6': (1038.0, 253.7),  # Swapped with M7
#         'M7': (1176.4, 238.8),  # Swapped with M6
#         'M8': (773.9, 271.2),   
#         'M9': (942.7, 330.1),   # Swapped with M10
#         'M10': (835.0, 335.6),  # Swapped with M11
#         'M11': (1100.8, 326.1), # Swapped with M9
#         'M12': (782.6, 445.2),  # Swapped with M13
#         'M13': (933.5, 464.2),  # Swapped with M14
#         'M14': (1187.9, 414.1), # Swapped with M12
#         'M15': (1050.4, 496.0), # Swapped with M16
#         'M16': (848.9, 519.7),  # Swapped with M17
#         'M17': (1137.3, 490.7), # Swapped with M15
#         'M18': (717.3, 804.1),  
#     },

#     "coor_3": {
#         'M0': (1182.5, 77.4),   
#         'M1': (755.3, 98.9),    
#         'M2': (1013.4, 107.0),  
#         'M3': (1101.1, 140.0),  
#         'M4': (927.6, 160.6),   # Swapped with M5
#         'M5': (822.3, 158.9),   # Swapped with M4
#         'M6': (1039.3, 250.5),  # Swapped with M7
#         'M7': (1178.4, 239.3),  # Swapped with M6
#         'M8': (775.3, 271.6),   
#         'M9': (944.1, 325.6),   
#         'M10': (838.6, 339.8),  # Swapped with M11
#         'M11': (1102.8, 330.9), # Swapped with M10
#         'M12': (784.6, 445.5),  # Swapped with M13
#         'M13': (934.1, 457.5),  # Swapped with M14
#         'M14': (1189.9, 415.6), # Swapped with M12
#         'M15': (1051.3, 489.2), 
#         'M16': (852.0, 525.5),  # Swapped with M17
#         'M17': (1139.0, 497.7), # Swapped with M16
#         'M18': (736.6, 862.8),  
#     },

# }

"""
Soft State 100g Near
"""
# REFERENCE_LAYOUTS = {

#     "coor_0": {
#         'M0': (1175.0, 86.0),
#         'M1': (750.1, 97.4),
#         'M2': (1008.5, 108.0),
#         'M3': (1093.0, 149.3),
#         'M4': (921.5, 160.1),
#         'M5': (816.7, 160.9),
#         'M6': (1029.6, 254.0),
#         'M7': (1166.8, 256.1),
#         'M8': (764.8, 268.1),
#         'M9': (930.7, 323.8),
#         'M10': (827.4, 344.2),
#         'M11': (1087.8, 352.4),
#         'M12': (764.2, 438.2),
#         'M13': (910.9, 451.7),   # Swapped with M14
#         'M14': (1167.8, 444.4),  # Swapped with M13
#         'M15': (1025.0, 494.7),
#         'M16': (827.0, 530.9),   # Swapped with M17
#         'M17': (1110.2, 530.2),  # Swapped with M16
#         'M18': (662.8, 807.2),  
#     },

#     "coor_1": {
#         'M0': (1177.2, 83.1),
#         'M1': (749.3, 94.1),
#         'M2': (1008.8, 110.2),
#         'M3': (1096.2, 142.1),
#         'M4': (921.7, 165.0),    # Swapped with M5
#         'M5': (815.1, 153.8),    # Swapped with M4
#         'M6': (1031.1, 262.7),   # Swapped with M7
#         'M7': (1169.3, 253.1),   # Swapped with M6
#         'M8': (762.7, 265.1),
#         'M9': (931.1, 336.5),    # Swapped with M10
#         'M10': (823.5, 330.0),   # Swapped with M9
#         'M11': (1090.4, 337.7),
#         'M12': (760.5, 435.6),
#         'M13': (910.9, 472.2),   # Swapped with M14
#         'M14': (1170.5, 440.2),  # Swapped with M13
#         'M15': (1026.5, 517.0),  # Swapped with M17
#         'M16': (822.3, 512.2),
#         'M17': (1113.9, 511.5),  # Swapped with M15
#         'M18': (654.2, 708.6), 
#     },

#     "coor_2": {
#         'M0': (1179.9, 75.2),
#         'M1': (751.9, 98.1),
#         'M2': (1009.5, 109.8),
#         'M3': (1099.9, 135.3),
#         'M4': (924.3, 167.2),    # Swapped with M5
#         'M5': (818.1, 155.9),    # Swapped with M4
#         'M6': (1037.8, 259.7),   # Swapped with M7
#         'M7': (1177.8, 237.3),   # Swapped with M6
#         'M8': (772.7, 276.5),
#         'M9': (943.1, 339.6),    # Swapped with M11
#         'M10': (836.7, 336.4),
#         'M11': (1103.8, 322.7),  # Swapped with M9
#         'M12': (785.1, 455.3),   # Swapped with M13
#         'M13': (935.2, 476.2),   # Swapped with M14
#         'M14': (1193.6, 412.3),  # Swapped with M12
#         'M15': (1055.9, 505.6),  # Swapped with M16
#         'M16': (856.3, 526.5),   # Swapped with M17
#         'M17': (1144.7, 487.9),  # Swapped with M15
#         'M18': (729.6, 793.8), 
#     },

#     "coor_3": {
#         'M0': (1179.0, 78.3),
#         'M1': (753.7, 100.9),
#         'M2': (1011.0, 106.2),
#         'M3': (1097.2, 143.2),
#         'M4': (925.3, 161.2),
#         'M5': (821.4, 163.7),
#         'M6': (1037.7, 251.1),   # Swapped with M7
#         'M7': (1176.7, 242.0),   # Swapped with M6
#         'M8': (775.1, 279.3),
#         'M9': (943.8, 327.9),
#         'M10': (840.1, 351.1),   # Swapped with M11
#         'M11': (1101.7, 339.3),  # Swapped with M10
#         'M12': (788.4, 458.6),   # Swapped with M13
#         'M13': (935.7, 460.4),   # Swapped with M14
#         'M14': (1191.6, 419.5),  # Swapped with M12
#         'M15': (1054.0, 488.9),
#         'M16': (860.4, 544.6),   # Swapped with M17
#         'M17': (1142.0, 508.0),  # Swapped with M16
#         'M18': (773.4, 927.2),
#     },

# }

"""
Stiff State 100g 
"""
# REFERENCE_LAYOUTS = {

#     "coor_0": {
#         'M0': (1148.0, 80.6),  # Swapped with Original M2
#         'M1': (758.9, 80.0),   
#         'M2': (999.9, 80.0),   # Swapped with Original M0
#         'M3': (1087.0, 188.5), # Was Original M4
#         'M4': (905.1, 191.2),  # Was Original M5
#         'M5': (801.7, 179.6),  # Was Original M3
#         'M6': (1019.8, 280.7), 
#         'M7': (1144.1, 283.6), 
#         'M8': (769.8, 288.8),  
#         'M9': (914.1, 364.6),  
#         'M10': (813.9, 400.5), # Swapped with Original M11
#         'M11': (1097.4, 399.0),# Swapped with Original M10
#         'M12': (766.2, 512.3), # Swapped with Original M14
#         'M13': (916.1, 504.6), 
#         'M14': (1141.0, 498.3),# Swapped with Original M12
#         'M15': (996.3, 586.9), 
#         'M16': (833.6, 633.8), # Swapped with Original M17
#         'M17': (1096.5, 613.1),# Swapped with Original M16
#         'M18': (649.4, 896.9), 
#     },

#     "coor_1": {
#         'M0': (1151.6, 72.8),  # Swapped with Original M2
#         'M1': (759.9, 77.7),   
#         'M2': (1005.3, 82.2),  # Swapped with Original M0
#         'M3': (1091.6, 178.2), # Was Original M4
#         'M4': (908.9, 196.4),  # Was Original M5
#         'M5': (802.9, 174.3),  # Was Original M3
#         'M6': (1027.0, 287.8), 
#         'M7': (1150.4, 273.1), 
#         'M8': (773.1, 284.6),  
#         'M9': (919.7, 373.6),  
#         'M10': (817.8, 392.1), # Swapped with Original M11
#         'M11': (1106.6, 386.8),# Swapped with Original M10
#         'M12': (770.9, 507.4), # Swapped with Original M14
#         'M13': (923.9, 516.7), 
#         'M14': (1151.6, 490.7),# Swapped with Original M12
#         'M15': (1007.2, 601.2), 
#         'M16': (840.3, 623.3), # Swapped with Original M17
#         'M17': (1109.4, 601.1),# Swapped with Original M16
#         'M18': (656.9, 777.5), 
#     },

#     "coor_2": {
#         'M0': (1158.3, 62.7),  # Swapped with Original M2
#         'M1': (765.4, 84.4),   
#         'M2': (1010.0, 79.5),  # Swapped with Original M0
#         'M3': (1101.9, 169.9), # Was Original M4
#         'M4': (917.9, 198.5),  # Was Original M5
#         'M5': (812.7, 180.2),  # Was Original M3
#         'M6': (1042.1, 282.9), 
#         'M7': (1166.9, 258.1), 
#         'M8': (789.5, 296.7),  
#         'M9': (939.7, 376.3),  
#         'M10': (841.2, 401.3), # Swapped with Original M11
#         'M11': (1129.7, 373.0),# Swapped with Original M10
#         'M12': (805.4, 524.1), # Swapped with Original M14
#         'M13': (957.0, 520.9), 
#         'M14': (1183.4, 470.1),# Swapped with Original M12
#         'M15': (1048.3, 596.7), 
#         'M16': (885.4, 633.5), # Swapped with Original M17
#         'M17': (1152.1, 583.0),# Swapped with Original M16
#         'M18': (741.9, 907.0), 
#     },

#     "coor_3": {
#         'M0': (1154.0, 71.8),  # Swapped with Original M2
#         'M1': (765.3, 86.2),   
#         'M2': (1006.0, 77.5),  # Swapped with Original M0
#         'M3': (1096.3, 181.2), # Was Original M4
#         'M4': (914.4, 194.7),  # Was Original M5
#         'M5': (811.0, 185.2),  # Was Original M3
#         'M6': (1034.9, 276.8), 
#         'M7': (1159.8, 267.8), 
#         'M8': (787.0, 302.0),  
#         'M9': (934.9, 368.1),  
#         'M10': (837.8, 411.4), # Swapped with Original M11
#         'M11': (1121.3, 386.4),# Swapped with Original M10
#         'M12': (803.0, 530.9), # Swapped with Original M14
#         'M13': (950.1, 509.3), 
#         'M14': (1174.1, 478.3),# Swapped with Original M12
#         'M15': (1038.3, 582.7), 
#         'M16': (880.3, 645.9), # Swapped with Original M17
#         'M17': (1141.7, 596.1),# Swapped with Original M16
#         'M18': (904.3, 1005.7),
#     },

# }

"""
Stiff State 20g
"""
# REFERENCE_LAYOUTS = {

#     "coor_0": {
#         'M0': (1152.4, 72.6),  
#         'M1': (760.5, 81.8),   # Swapped with Original M2
#         'M2': (1004.0, 78.1),  # Swapped with Original M1
#         'M3': (1093.0, 180.1), # Was Original M4
#         'M4': (910.5, 193.1),  # Was Original M5
#         'M5': (804.4, 179.8),  # Was Original M3
#         'M6': (1029.4, 278.1), # Swapped with Original M7
#         'M7': (1154.3, 269.4), # Swapped with Original M6
#         'M8': (777.4, 291.4),  
#         'M9': (926.8, 366.8),  
#         'M10': (825.2, 400.0), # Swapped with Original M11
#         'M11': (1112.8, 385.1),# Swapped with Original M10
#         'M12': (783.8, 515.4), # Swapped with Original M14
#         'M13': (935.6, 507.6), 
#         'M14': (1161.3, 481.5),# Swapped with Original M12
#         'M15': (1020.1, 585.3), 
#         'M16': (857.2, 631.3), # Swapped with Original M17
#         'M17': (1123.7, 595.6),# Swapped with Original M16
#         'M18': (817.3, 910.1), 
#     },

#     "coor_1": {
#         'M0': (1153.1, 72.7),  
#         'M1': (760.6, 79.6),   # Swapped with Original M2
#         'M2': (1004.8, 79.4),  # Swapped with Original M1
#         'M3': (1093.4, 178.8), # Was Original M4
#         'M4': (910.3, 194.1),  # Was Original M5
#         'M5': (804.3, 177.3),  # Was Original M3
#         'M6': (1029.0, 280.4), # Swapped with Original M7
#         'M7': (1154.2, 268.7), # Swapped with Original M6
#         'M8': (776.5, 289.0),  
#         'M9': (925.2, 368.5),  
#         'M10': (823.7, 396.7), # Swapped with Original M11
#         'M11': (1112.1, 383.5),# Swapped with Original M10
#         'M12': (781.6, 513.0), # Swapped with Original M14
#         'M13': (933.5, 510.3), 
#         'M14': (1160.1, 481.6),# Swapped with Original M12
#         'M15': (1017.8, 589.1), 
#         'M16': (854.4, 627.9), # Swapped with Original M17
#         'M17': (1122.5, 594.2),# Swapped with Original M16
#         'M18': (803.4, 879.9), 
#     },

#     "coor_2": {
#         'M0': (1153.8, 68.8),  
#         'M1': (760.2, 80.3),   # Swapped with Original M2
#         'M2': (1005.1, 77.8),  # Swapped with Original M1
#         'M3': (1094.5, 176.0), 
#         'M4': (911.6, 193.5),  # Swapped with Original M5
#         'M5': (804.7, 178.1),  # Swapped with Original M4
#         'M6': (1031.9, 279.1), # Swapped with Original M7
#         'M7': (1157.1, 264.6), # Swapped with Original M6
#         'M8': (778.4, 290.9),  
#         'M9': (929.1, 369.0),  
#         'M10': (827.4, 398.3), # Swapped with Original M11
#         'M11': (1116.2, 380.1),# Swapped with Original M10
#         'M12': (787.3, 516.2), # Swapped with Original M14
#         'M13': (939.9, 511.2), 
#         'M14': (1166.9, 476.7),# Swapped with Original M12
#         'M15': (1026.7, 588.0), 
#         'M16': (862.9, 630.2), # Swapped with Original M17
#         'M17': (1130.6, 590.5),# Swapped with Original M16
#         'M18': (822.2, 889.7), 
#     },

#     "coor_3": {
#         'M0': (1152.6, 72.8),  
#         'M1': (761.0, 83.6),   # Swapped with Original M2
#         'M2': (1004.2, 79.1),  # Swapped with Original M1
#         'M3': (1093.2, 179.6), 
#         'M4': (911.7, 194.8),  # Swapped with Original M5
#         'M5': (805.7, 180.7),  # Swapped with Original M4
#         'M6': (1030.7, 278.2), # Swapped with Original M7
#         'M7': (1155.5, 268.2), # Swapped with Original M6
#         'M8': (779.4, 293.5),  
#         'M9': (928.9, 367.3),  
#         'M10': (828.3, 401.0), # Swapped with Original M11
#         'M11': (1114.5, 383.2),# Swapped with Original M10
#         'M12': (788.2, 517.7), # Swapped with Original M14
#         'M13': (939.5, 508.2), 
#         'M14': (1164.7, 478.3),# Swapped with Original M12
#         'M15': (1025.2, 585.0), 
#         'M16': (863.7, 631.8), # Swapped with Original M17
#         'M17': (1128.7, 592.1),# Swapped with Original M16
#         'M18': (840.8, 937.4), 
#     },

# }

"""
Mix State 20g
"""
# REFERENCE_LAYOUTS = {

#     "coor_0": {
#         'M0': (1168.5, 66.6),
#         'M1': (762.5, 84.7),   # Swapped with Original M2
#         'M2': (1005.2, 74.0),  # Swapped with Original M1
#         'M3': (1096.6, 143.9),
#         'M4': (924.3, 167.1),
#         'M5': (815.3, 183.6),
#         'M6': (1034.0, 270.1),
#         'M7': (1180.9, 270.5),
#         'M8': (781.3, 292.4),
#         'M9': (936.0, 361.5),  # Swapped with Original M10
#         'M10': (829.6, 392.6), # Swapped with Original M11
#         'M11': (1121.7, 341.5),# Swapped with Original M9
#         'M12': (805.8, 496.2), # Swapped with Original M13
#         'M13': (954.9, 503.3), # Swapped with Original M14
#         'M14': (1205.2, 461.4),# Swapped with Original M12
#         'M15': (1046.0, 569.6),# Swapped with Original M16
#         'M16': (870.5, 610.1), # Swapped with Original M17
#         'M17': (1147.9, 546.0),# Swapped with Original M15
#         'M18': (887.4, 884.7), 
#     },

#     "coor_1": {
#         'M0': (1168.3, 67.5),
#         'M1': (762.7, 83.9),   # Swapped with Original M2
#         'M2': (1004.3, 75.5),  # Swapped with Original M1
#         'M3': (1096.3, 143.1),
#         'M4': (923.8, 168.7),
#         'M5': (814.7, 182.0),
#         'M6': (1033.0, 271.7), # Swapped with Original M7
#         'M7': (1180.4, 270.2), # Swapped with Original M6
#         'M8': (780.9, 291.2),
#         'M9': (935.0, 363.3),  # Swapped with Original M10
#         'M10': (828.4, 390.0), # Swapped with Original M11
#         'M11': (1121.1, 338.1),# Swapped with Original M9
#         'M12': (805.4, 494.5), # Swapped with Original M13
#         'M13': (954.0, 504.3), # Swapped with Original M14
#         'M14': (1204.6, 459.4),# Swapped with Original M12
#         'M15': (1044.9, 571.2),# Swapped with Original M16
#         'M16': (869.3, 606.1), # Swapped with Original M17
#         'M17': (1147.4, 541.1),# Swapped with Original M15
#         'M18': (877.1, 851.4),
#     },

#     "coor_2": {
#         'M0': (1169.3, 65.4),
#         'M1': (763.6, 84.9),   # Swapped with Original M2
#         'M2': (1004.7, 74.9),  # Swapped with Original M1
#         'M3': (1098.5, 140.7),
#         'M4': (924.9, 168.9),
#         'M5': (816.5, 182.4),
#         'M6': (1035.4, 270.2), # Swapped with Original M7
#         'M7': (1183.4, 265.3), # Swapped with Original M6
#         'M8': (784.2, 293.0),
#         'M9': (938.9, 363.2),  # Swapped with Original M10
#         'M10': (833.4, 390.7), # Swapped with Original M11
#         'M11': (1125.5, 333.0),# Swapped with Original M9
#         'M12': (812.8, 497.4), # Swapped with Original M13
#         'M13': (961.4, 504.4), # Swapped with Original M14
#         'M14': (1210.8, 451.0),# Swapped with Original M12
#         'M15': (1054.2, 569.1),# Swapped with Original M16
#         'M16': (879.2, 605.8), # Swapped with Original M17
#         'M17': (1156.1, 532.2),# Swapped with Original M15
#         'M18': (905.7, 861.0), 
#     },

#     "coor_3": {
#         'M0': (1168.4, 66.7),
#         'M1': (763.4, 86.1),   # Swapped with Original M2
#         'M2': (1003.6, 75.1),  # Swapped with Original M1
#         'M3': (1097.8, 143.5),
#         'M4': (924.0, 168.4),
#         'M5': (816.5, 184.5),
#         'M6': (1033.8, 270.1), # Swapped with Original M7
#         'M7': (1182.1, 268.0), # Swapped with Original M6
#         'M8': (783.8, 294.4),
#         'M9': (937.2, 362.6),  # Swapped with Original M10
#         'M10': (832.5, 393.1), # Swapped with Original M11
#         'M11': (1124.5, 337.9),# Swapped with Original M9
#         'M12': (811.8, 499.7), # Swapped with Original M13
#         'M13': (958.9, 503.8), # Swapped with Original M14
#         'M14': (1208.5, 454.5),# Swapped with Original M12
#         'M15': (1051.2, 568.5),# Swapped with Original M16
#         'M16': (878.2, 609.8), # Swapped with Original M17
#         'M17': (1154.4, 538.3),# Swapped with Original M15
#         'M18': (930.7, 917.0), 
#     },

# }

"""
Soft State No load
"""
REFERENCE_LAYOUTS = {

    "coor_0": {
        'M0': (1182.7, 72.4),  # Was Original M0
        'M1': (758.4, 86.0),   # Swapped with Original M2
        'M2': (1023.9, 85.6),  # Swapped with Original M1
        'M3': (1107.1, 147.6), # Was Original M3
        'M4': (922.4, 153.4),  # Swapped with Original M5
        'M5': (819.3, 149.2),  # Swapped with Original M4
        'M6': (1042.5, 259.9), # Swapped with Original M7
        'M7': (1193.4, 255.0), # Swapped with Original M6
        'M8': (768.2, 275.2),  # Was Original M8
        'M9': (934.3, 330.6),  # Swapped with Original M10
        'M10': (836.1, 338.3), # Swapped with Original M11
        'M11': (1122.9, 315.5),# Swapped with Original M9
        'M12': (768.0, 454.6), # Swapped with Original M13
        'M13': (932.4, 457.0), # Swapped with Original M14
        'M14': (1182.8, 418.3),# Swapped with Original M12
        'M15': (1030.8, 513.5),# Was Original M15
        'M16': (847.2, 521.3), # Was Original M16
        'M17': (1118.1, 527.5),# Was Original M17
        'M18': (809.8, 813.5), # Was Original M18
    },

}

# Each tuple: (angle_key, source_template, hdf5_template, n_samples)
# Use {i} as placeholder for sample_id in the templates.
TRIAL_BATCHES = [
    ("coor_0", f"{BASE}/soft_state_noload/coor_0/C1412_sample_{{i}}.mp4", f"{BASE}/soft_state_noload/coor_0/trajectories_sample_{{i}}.h5", 12),
    # ("coor_1", f"{BASE}/mix_state_20g/coor_1/C1393_sample_{{i}}.mp4", f"{BASE}/mix_state_20g/coor_1/trajectories_sample_{{i}}.h5", 12),
    # ("coor_2", f"{BASE}/mix_state_20g/coor_2/C1394_sample_{{i}}.mp4", f"{BASE}/mix_state_20g/coor_2/trajectories_sample_{{i}}.h5", 12),
    # ("coor_3", f"{BASE}/mix_state_20g/coor_3/C1395_sample_{{i}}.mp4", f"{BASE}/mix_state_20g/coor_3/trajectories_sample_{{i}}.h5", 12),
]

# ═══════════════════════════════════════════════════════════════


def _check_layouts():
    errors = []
    for angle_key, layout in REFERENCE_LAYOUTS.items():
        if len(layout) == 0:
            errors.append(f"  '{angle_key}' — empty.")
        else:
            missing = [f"M{i}" for i in range(NUM_MARKERS) if f"M{i}" not in layout]
            if missing:
                errors.append(f"  '{angle_key}' — missing keys: {missing}")
    for angle_key, _, _, _ in TRIAL_BATCHES:
        if angle_key not in REFERENCE_LAYOUTS:
            errors.append(f"  TRIAL_BATCHES references '{angle_key}' not in REFERENCE_LAYOUTS.")
    if errors:
        print("\n[ERROR] REFERENCE_LAYOUTS incomplete:")
        for e in errors: print(e)
        sys.exit(1)
    print("[INFO] All reference layouts validated.\n")


def _layout_to_list(layout):
    return [layout[f"M{i}"] for i in range(NUM_MARKERS)]


@dataclass
class Marker:
    marker_id:   int
    centroid:    Tuple[float, float]
    area:        float
    bbox:        Tuple[int, int, int, int]
    contour:     np.ndarray
    history:     deque = field(default_factory=lambda: deque(maxlen=60))
    lost_frames: int   = 0
    active:      bool  = True
    kalman:      Optional[cv2.KalmanFilter] = None

    def __post_init__(self):
        self.history.append(self.centroid)
        self.kalman = self._init_kalman(self.centroid)

    def _init_kalman(self, centroid):
        kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf.processNoiseCov       = np.eye(4, dtype=np.float32) * 1e-1
        kf.processNoiseCov[2, 2] = 5.0
        kf.processNoiseCov[3, 3] = 5.0
        kf.measurementNoiseCov   = np.eye(2, dtype=np.float32) * 1e-2
        kf.errorCovPost          = np.eye(4, dtype=np.float32)
        kf.statePost = np.array(
            [centroid[0], centroid[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        return kf

    def predict(self):
        p = self.kalman.predict()
        return float(p[0][0]), float(p[1][0])

    def correct(self, centroid):
        self.kalman.correct(np.array([[centroid[0]], [centroid[1]]], dtype=np.float32))
        self.centroid    = centroid
        self.lost_frames = 0
        self.active      = True
        self.history.append(centroid)

    def update_lost(self):
        pred = self.predict()
        self.centroid = pred
        self.history.append(pred)
        self.lost_frames += 1


def _morph_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def detect_candidates(frame):
    bk   = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
    mk   = _morph_kernel(MORPH_SIZE)
    blur = cv2.GaussianBlur(frame, (bk, bk), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array(RED_LOWER_1), np.array(RED_UPPER_1)),
        cv2.inRange(hsv, np.array(RED_LOWER_2), np.array(RED_UPPER_2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  mk, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mk, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA): continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        perim = cv2.arcLength(cnt, True)
        if perim == 0 or 4 * np.pi * area / (perim * perim) < 0.1: continue
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append({"centroid": (cx, cy), "area": area,
                            "bbox": (x, y, w, h), "contour": cnt})
    return candidates, mask


def match_to_markers(candidates, markers):
    """
    Build cost matrix from Kalman PREDICTED positions, not fixed reference.
    Follows the marker trajectory during oscillation so fast-moving markers
    don't exceed MAX_DIST and go undetected.
    Reference is only used for initialisation and lost-marker recovery.
    """
    N, M = len(markers), len(candidates)
    if M == 0: return {}
    cost = np.full((N, M), MAX_DIST * 10, dtype=np.float64)
    for i, marker in enumerate(markers):
        px, py = marker.predict()
        for j, det in enumerate(candidates):
            d = np.hypot(px - det["centroid"][0], py - det["centroid"][1])
            if d < MAX_DIST: cost[i, j] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    return {r: candidates[c] for r, c in zip(row_ind, col_ind) if cost[r, c] < MAX_DIST}


class RedMarkerTracker:
    def __init__(self, reference):
        self.reference = reference   # kept for lost-marker recovery only
        self.markers = [
            Marker(marker_id=mid, centroid=pos, area=0.0,
                   bbox=(0,0,0,0), contour=np.zeros((1,1,2), dtype=np.int32))
            for mid, pos in enumerate(reference)
        ]
        self.fps_history = deque(maxlen=30)
        self._prev_time  = time.time()

    def process_frame(self, frame):
        now = time.time()
        dt  = now - self._prev_time
        self._prev_time = now
        self.fps_history.append(1.0 / dt if dt > 0 else 0.0)
        fps = float(np.mean(self.fps_history))

        original = frame.copy()
        candidates, mask = detect_candidates(frame)
        assignment = match_to_markers(candidates, self.markers)

        for m in self.markers:
            if m.marker_id in assignment:
                det = assignment[m.marker_id]
                m.area, m.bbox, m.contour = det["area"], det["bbox"], det["contour"]
                m.correct(det["centroid"])
            else:
                m.update_lost()
                if m.lost_frames > MAX_LOST // 2:
                    m.correct(self.reference[m.marker_id])
                    m.lost_frames = max(0, m.lost_frames - 1)
                if m.lost_frames > MAX_LOST:
                    m.active = False

        overlay = original.copy()
        black   = np.zeros(frame.shape, dtype=np.uint8)
        GREEN   = (0, 255, 0)
        for m in self.markers:
            if not m.active: continue
            cx, cy = int(m.centroid[0]), int(m.centroid[1])
            cv2.circle(overlay, (cx, cy), 8, GREEN, -1)
            cv2.circle(overlay, (cx, cy), 8, (0,0,0), 1)
            if m.lost_frames == 0:
                cv2.drawContours(overlay, [m.contour], -1, GREEN, 1)
            cv2.circle(black, (cx, cy), 8, GREEN, -1)
            cv2.circle(black, (cx, cy), 8, (255,255,255), 1)
            if SHOW_IDS:
                label = f"M{m.marker_id}"
                cv2.putText(overlay, label, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
                cv2.putText(overlay, label, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
                cv2.putText(black,   label, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        n_active = sum(1 for m in self.markers if m.active)
        hud = f"Tracking: {n_active}/{NUM_MARKERS}   FPS: {fps:.1f}"
        for img in (overlay, black):
            cv2.putText(img, hud, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
        return original, mask, overlay, black


class TrajectoryStore:
    def __init__(self, video_fps):
        self.video_fps = video_fps
        self._times, self._pos = [], []

    def record(self, frame_idx, time_s, markers):
        self._times.append(time_s)
        row = [None] * NUM_MARKERS
        for m in markers:
            if 0 <= m.marker_id < NUM_MARKERS and m.active:
                row[m.marker_id] = m.centroid
        self._pos.append(row)

    def save_h5(self, source, path):
        F, N = len(self._times), NUM_MARKERS
        if F == 0:
            print("[WARN] No frames recorded — HDF5 not written.")
            return
        pos_arr = np.full((F, N, 2), np.nan, dtype=np.float64)
        for fi, row in enumerate(self._pos):
            for mid, xy in enumerate(row):
                if xy is not None:
                    pos_arr[fi, mid] = xy
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(p), "w") as f:
            f.attrs["source_file"] = str(source)
            f.attrs["video_fps"]   = self.video_fps
            f.attrs["num_markers"] = N
            f.attrs["num_frames"]  = F
            ts = f.create_group("time_series")
            ts.create_dataset("time", data=np.array(self._times), compression="gzip")
            nodes = ts.create_group("nodes")
            ds = nodes.create_dataset("positions", data=pos_arr, compression="gzip")
            ds.attrs["units"] = "pixels"
            ds.attrs["axes"]  = "frame marker xy"
            nodes.create_dataset("node_ids", data=np.arange(N, dtype=np.int32))
        print(f"[INFO] HDF5 saved → {p.resolve()}  shape={pos_arr.shape}")

    def print_summary(self):
        F = len(self._times)
        if F == 0: return
        print("\n" + "="*50)
        print(f"  Frames: {F}   Duration: {self._times[-1]:.3f}s")
        for mid in range(NUM_MARKERS):
            det = sum(1 for row in self._pos if row[mid] is not None)
            print(f"  M{mid:>2}  {det:>6}/{F}  {100*det/F:>6.1f}%")
        print("="*50 + "\n")


def draw_quad_view(original, mask, overlay, black):
    hw, hh = QUAD_W // 2, QUAD_H // 2
    def r(img): return cv2.resize(img, (hw, hh), interpolation=cv2.INTER_LINEAR)
    return np.vstack([
        np.hstack([r(original), r(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))]),
        np.hstack([r(overlay),  r(black)]),
    ])


def add_labels(canvas):
    hw, hh = QUAD_W // 2, QUAD_H // 2
    for (x, y), text in [
        ((8,      hh-8), "Original"),
        ((hw+8,   hh-8), "HSV Mask"),
        ((8,      QUAD_H-8), "Tracked (overlay)"),
        ((hw+8,   QUAD_H-8), "Tracked (black bg)"),
    ]:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.line(canvas, (hw, 0), (hw, QUAD_H), (60,60,60), 1)
    cv2.line(canvas, (0, hh), (QUAD_W, hh), (60,60,60), 1)


def hsv_tuner(cap):
    WIN = "HSV Tuner  (Q = done)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 600)
    def nothing(_): pass
    for name, val, mx in [
        ("H_lo1",0,30),("S_lo1",75,255),("V_lo1",80,255),
        ("H_hi1",10,30),("S_hi1",255,255),("V_hi1",255,255),
        ("H_lo2",165,180),("S_lo2",75,255),("V_lo2",80,255),
        ("H_hi2",180,180),("S_hi2",255,255),("V_hi2",255,255),
    ]:
        cv2.createTrackbar(name, WIN, val, mx, nothing)
    mk = _morph_kernel(3)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read()
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5,5), 0), cv2.COLOR_BGR2HSV)
        def tb(n): return cv2.getTrackbarPos(n, WIN)
        lo1 = np.array([tb("H_lo1"),tb("S_lo1"),tb("V_lo1")], dtype=np.uint8)
        hi1 = np.array([tb("H_hi1"),tb("S_hi1"),tb("V_hi1")], dtype=np.uint8)
        lo2 = np.array([tb("H_lo2"),tb("S_lo2"),tb("V_lo2")], dtype=np.uint8)
        hi2 = np.array([tb("H_hi2"),tb("S_hi2"),tb("V_hi2")], dtype=np.uint8)
        mask = cv2.morphologyEx(
            cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2)),
            cv2.MORPH_CLOSE, mk, iterations=2)
        cv2.imshow(WIN, np.hstack([cv2.resize(frame,(450,400)),
                                   cv2.resize(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),(450,400))]))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("\n--- Update CONFIG in this script ---")
            print(f"RED_LOWER_1 = {tuple(lo1.tolist())}")
            print(f"RED_UPPER_1 = {tuple(hi1.tolist())}")
            print(f"RED_LOWER_2 = {tuple(lo2.tolist())}")
            print(f"RED_UPPER_2 = {tuple(hi2.tolist())}")
            break
    cv2.destroyWindow(WIN)


def process_trial(source, hdf5_output, reference, video_fps):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); return
    fps = cap.get(cv2.CAP_PROP_FPS) or video_fps

    tracker = RedMarkerTracker(reference)
    store   = TrajectoryStore(fps)

    WIN = "Red Marker Tracker  [Q=quit | S=screenshot | T=HSV tuner]"
    if SHOW_PREVIEW:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, QUAD_W, QUAD_H)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            time_s = frame_idx / fps
            original, mask, overlay, black = tracker.process_frame(frame)
            store.record(frame_idx, time_s, tracker.markers)
            if SHOW_PREVIEW:
                canvas = draw_quad_view(original, mask, overlay, black)
                add_labels(canvas)
                cv2.imshow(WIN, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('s'):
                    fn = f"screenshot_{frame_idx:05d}.png"
                    cv2.imwrite(fn, canvas); print(f"[INFO] Screenshot: {fn}")
                elif key == ord('t'):
                    hsv_tuner(cap)
            if frame_idx % 100 == 0 and frame_idx > 0:
                n = sum(1 for m in tracker.markers if m.active)
                print(f"  frame {frame_idx:6d}  t={time_s:.4f}s  active: {n}/{NUM_MARKERS}")
            frame_idx += 1
    finally:
        cap.release()
        if SHOW_PREVIEW: cv2.destroyAllWindows()

    store.print_summary()
    if hdf5_output:
        store.save_h5(source, hdf5_output)


if __name__ == "__main__":
    _check_layouts()

    for angle_key, src_tmpl, hdf5_tmpl, n_samples in TRIAL_BATCHES:
        reference = _layout_to_list(REFERENCE_LAYOUTS[angle_key])
        print(f"\n{'='*60}")
        print(f"  Condition : {angle_key}  ({n_samples} trials)")
        print(f"{'='*60}")
        for i in range(n_samples):
            source = src_tmpl.replace("{i}", str(i))
            hdf5   = hdf5_tmpl.replace("{i}", str(i))
            print(f"\n  Trial {i:2d}/{n_samples-1}  —  {Path(source).name}")
            process_trial(source, hdf5, reference, VIDEO_FPS)

    print("\n[DONE] All conditions and trials processed.")