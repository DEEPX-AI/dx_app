import subprocess
import os
import time
cmd = "./bin/segmentation -m assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -i sample/8.jpg"

complete_cnt = 0
failure_cnt = 0
retry_count = 0
i = 0
while i < 10:
    result = subprocess.Popen(cmd.split(" "), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, _ = result.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        print("[dx-app notice] timeout. terminate process!!")
        print("restart test .... after 3 seconds")
        time.sleep(3)
        retry_count += 1
        if retry_count >= 3:
            print("maximum retry count reached. exit test.")
            result.terminate()
            stdout, _ = result.communicate()
        else: 
            continue
    if "Result saved to result.jpg" in stdout :
        print(str(__file__) + " complete!!")
        complete_cnt += 1
    else:
        print(str(__file__) + " failure!!")
        print("************* LOG PRINT *************")
        print(stdout)
        failure_cnt += 1
    i += 1

print("[dx-app script test] {}, COMPLETE : {}, FAILURE : {}".format(os.path.basename(__file__), complete_cnt, failure_cnt))

if failure_cnt > 0:
    exit(-1)
else:
    exit(0)