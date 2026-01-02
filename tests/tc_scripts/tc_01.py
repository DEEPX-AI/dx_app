import subprocess
import os
import time
cmd = "./bin/classification -m ./assets/models/EfficientNetB0_4.dxnn -i ./sample/ILSVRC2012/1.jpeg"

complete_cnt = 0
failure_cnt = 0
for i in range(10):
    result = subprocess.Popen(cmd.split(" "), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, _ = result.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        result.terminate()
        stdout, _ = result.communicate()
    if "Top1 Result : class 321" in stdout:
        print(str(__file__) + " complete!!")
        complete_cnt += 1
    else:
        print(str(__file__) + " failure!!")
        print("************* LOG PRINT *************")
        print(stdout)
        failure_cnt += 1

print("[dx-app script test] {}, COMPLETE : {}, FAILURE : {}".format(os.path.basename(__file__), complete_cnt, failure_cnt))

if failure_cnt > 0:
    exit(-1)
else:
    exit(0)