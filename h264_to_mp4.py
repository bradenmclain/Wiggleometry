import os 
import glob

extension = '.h264'
files = glob.glob('*.{}'.format(extension))

for file in files:
    print(file)