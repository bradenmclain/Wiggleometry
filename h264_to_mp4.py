import os 
import glob
import subprocess

extension = '.h264'
cwd = os.getcwd()
files = (glob.glob(cwd+'/*.h264'))


for file in files:
    filename = file.split('/')[-1]
    new_filename = filename.split('.')[0]
    new_filename += '.mp4'

    command = f'ffmpeg -i {filename} -c copy {new_filename}'

    completed_process = subprocess.run(command, shell=True, check=True)
    return_code = completed_process.returncode

    print(f'command returned {return_code}')

    
