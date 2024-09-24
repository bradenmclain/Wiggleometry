import json
import os
import sys
import glob

path = sys.argv[1]

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

for filename in glob.glob(os.path.join(path, '*_data.txt')):
    with open(filename, 'r') as f:
    
        name = filename.split('_data')[0].split('/')[2]


        data = json.load(f)

        for laser_power, bead_data in videos.items():
            if name in bead_data:
                # Update the bead data in the dictionary
                # videos[laser_power][name][1] = data['stats']['XY error average']
                # videos[laser_power][name][2] = data['stats']['Z error average']
                videos[laser_power][name][1] = (data['stats']["Z Norm Avg"])
                videos[laser_power][name][2] = (data['stats']["XY Norm stdev"])
                videos[laser_power][name][3] = (data['stats']["Z Norm stdev"])

            

                #print(f"Updated {name} in video {video_id} with data: {data}")

for laser_power, beads in videos.items():
    print(f"Laser Power: {laser_power}")
    for bead, data in beads.items():
        #print(f"  Bead {bead}: Unstable Time: {data[0]}\tXY error average: {data[1]}\tZ error average: {data[2]} XY Norm: {data[3]}")
        print(f"  Bead {bead}: Unstable Time: {data[0]} Z Norm Avg {data[1]} Z Norm stdev {data[3]}")
        # print(data['stats']['XY error range'])

        # print(data['stats']['XY error stdev'])
        # print(data['stats']['Z error range'])

        # print(data['stats']['Z error stdev'])

        #print(f"XY error average for sample {name} is {data['stats']['XY error average']}")
        #print(f"Z error range for sample {name} is {data['stats']['Z error range']}")
        #print(f"Z error average for sample {name} is {data['stats']['Z error average']}\n")
        