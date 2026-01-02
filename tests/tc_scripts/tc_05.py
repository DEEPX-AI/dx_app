import subprocess
import os, time
import re
cmd = "./bin/od_segmentation -m0 ./assets/models/YOLOV5S_3.dxnn -p0 1 -m1 ./assets/3148/HyundaiDDRNet_1.dxnn -p1 1 -i sample/7.jpg"

complete_cnt = 0
failure_cnt = 0
for i in range(10):
    result = subprocess.Popen(cmd.split(" "), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, _ = result.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        result.terminate()
        stdout, _ = result.communicate()
    found_bbox = re.search(r"Detected (\d+) boxes.", stdout)
    if "Result saved to result.jpg" in stdout and int(found_bbox.group(1)) == 3:
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