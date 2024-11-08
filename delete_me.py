import requests
import time

start = time.time()
x = requests.get('http://0.0.0.0:8080/status')
print(f'it took {time.time()-start}')
print(x.status_code)