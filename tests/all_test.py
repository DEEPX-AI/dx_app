import subprocess
import os
import threading

scripts_dir = "./test/tc_scripts/"
tcs = os.listdir(scripts_dir)
P_GREEN_S = "\033[32m"
P_GREEN_E = "\033[0m"

P_RED_S = "\033[31m"
P_RED_E = "\033[0m"

def subprocess_run(cmd, idx):
    result = subprocess.run(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(P_GREEN_S + cmd + " : PASSED [{}]".format(idx) + P_GREEN_E)
    else:
        print(P_RED_S + cmd + " : FAILED [{}]".format(idx) + P_RED_E)

threads = []
for l in range(2):
    for tc in tcs:
        cmd = "python " + scripts_dir + tc
        threads.append(threading.Thread(target=subprocess_run, args=[cmd, l]))
    
for thr in threads:
    thr.start()
    
for thr in threads:
    thr.join()
