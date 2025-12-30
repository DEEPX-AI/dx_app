import subprocess
import os
import time
cmd = "./bin/run_detector -c ./test/data/camera_test.json"

complete_cnt = 0
failure_cnt = 0

process = subprocess.Popen(cmd.split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

for i in range(10):
    time.sleep(1) 
    print("[{} sec] Waiting...".format(i + 1))

process.stdin.write("q\n")
process.stdin.flush()

try:
    stdout, _ = process.communicate(timeout=2)
except subprocess.TimeoutExpired:
    process.terminate()
    stdout, _ = process.communicate()

if "[DX-APP ERROR] The camera capture frame did not arrive properly." in stdout :
    print(str(__file__) + " complete!!")
    complete_cnt += 1
else:
    print(str(__file__) + " failure!!")
    print("************* LOG PRINT *************")
    print(stdout)
    failure_cnt += 1

if process.returncode == 0:
    print(str(__file__) + " closing process complete!!")
else:
    print(str(__file__) + " closing process failure!!")

print("[dx-app script test] {}, COMPLETE : {}, FAILURE : {}".format(os.path.basename(__file__), complete_cnt, failure_cnt))

if failure_cnt > 0:
    exit(-1)
else:
    exit(0)