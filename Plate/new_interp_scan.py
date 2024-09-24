import json
import os
import sys
from collections import OrderedDict
import numpy as np

videos = {  
    1500:   
        {'bead_0': [100,0,0,0],
        'bead_6': [100,0,0,0],
        'bead_7': [100,0,0,0],
        },
    1450:   
        {'bead_8': [0.0,0,0,0],
        'bead_9': [100,0,0,0],
        'bead_10': [0.0,0,0,0],
        },
    1400:   
        {'bead_1': [12.5,0,0,0],
        'bead_30': [0.0,0,0,0],
        'bead_12': [0.0,0,0,0],
        },
    1350:   
        {'bead_13': [0,0,0,0],
        'bead_14': [5.92,0,0,0],
        'bead_15': [0,0,0,0],
        },
    1300:   
        {'bead_16': [15.32,0,0,0],
        'bead_17': [6.84,0,0,0],
        'bead_18': [8.78,0,0,0],
        },
    1250:   
        {'bead_19': [23.23,0,0,0],
        'bead_20': [0,0,0,0],
        'bead_21': [0.73,0,0,0],
        },
    1200:   
        {'bead_22': [6.94,0,0,0],
        'bead_23': [1.43,0,0,0],
        'bead_24': [0.74,0,0,0],
        },
    1150:   
        {'bead_25': [11.02,0,0,0],
        'bead_26': [24.44,0,0,0],
        'bead_27': [88.65,0,0,0],
        },
    1100:   
        {'bead_28': [96.57,0,0,0],
        'bead_29': [91.48,0,0,0],
        'bead_5': [82.51,0,0,0],
        },
}

if __name__=="__main__":
    if os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], 'r') as f:
            name = sys.argv[1].split('_')[0]
            data= json.load(f)
            if name == '6-13':
                nums = np.linspace(13,6,8)
            if name == '23-30':
                nums = np.linspace(23,30,8)
            if name == '14-22':
                nums = np.linspace(22,14,9)
            if name == '0':
                nums = [0,1,5]



        sorted_beads = sorted(data['stats'].keys(), key=lambda x: float(x),reverse=False)

        # Create a sorted dictionary based on the float-sorted keys
        sort_beads = {i: data['stats'][i] for i in sorted_beads}

        for idx,bead in enumerate(sort_beads):
            print(f"bead {str(nums[idx])} has XY Average {data['stats'][bead]['XY error average']} and Z Average {data['stats'][bead]['Z error average']}")
            
        for bead in sort_beads:
            #print(f"bead {bead} has Z Average {data['stats'][bead]['Z error average']}")
            pass