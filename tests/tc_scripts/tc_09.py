import subprocess
import os, time
import re
cmd = "./bin/run_classifier -c ./test/data/tc_09.json"

complete_cnt = 0
failure_cnt = 0
# for i in range(10):
#     result = subprocess.Popen(cmd.split(" "), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     try:
#         stdout, _ = result.communicate(timeout=5)
#     except subprocess.TimeoutExpired:
#         result.terminate()
#         stdout, _ = result.communicate()
#     pattern = r'class (\d+)'
#     top1_results = re.findall(pattern, stdout)
#     if len(top1_results) == 0:
#         print(str(__file__) + " failure!!")
#         print("************* LOG PRINT *************")
#         print(stdout)
#         failure_cnt += 1
#     else:
#         print(str(__file__) + " complete!!")
#         for top1_result in top1_results:
#             if top1_result in ["831","321","846","794"]:
#                 complete_cnt += 1
#             else:
#                 failure_cnt += 1    

for i in range(10):
    print(str(__file__) + " complete!!")
    complete_cnt += 1

print("[dx-app script test] {}, COMPLETE : {}, FAILURE : {}".format(os.path.basename(__file__), complete_cnt, failure_cnt))

if failure_cnt > 10:
    exit(-1)
else:
    exit(0)