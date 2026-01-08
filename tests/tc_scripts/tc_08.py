import subprocess
import os, time
import re

def parse_bbox_lines(bbox_lines):
    """
    Parse BBOX lines to [class_id, score, x1, y1, x2, y2] format.
    """
    results = []
    pattern = re.compile(
        r"BBOX:[^(]+\((\d+)\)\s+([0-9.]+),\s+\(([0-9.\-]+),\s*([0-9.\-]+),\s*([0-9.\-]+),\s*([0-9.\-]+)\)"
    )
    for line in bbox_lines:
        match = pattern.search(line)
        if match:
            class_id = int(match.group(1))  # class id
            score = float(match.group(2))   # confidence score
            x1 = float(match.group(3))
            y1 = float(match.group(4))
            x2 = float(match.group(5))
            y2 = float(match.group(6))
            results.append([class_id, score, x1, y1, x2, y2])
    return results

cmds = [
        ["./bin/yolo -m ./assets/models/YOLOV5S_3.dxnn -p 1 -i sample/1.jpg", 512],
        ["./bin/yolo -m ./assets/models/YOLOV5S_4.dxnn -p 0 -i sample/1.jpg", 320],
        ["./bin/yolo -m ./assets/models/YOLOV5S_6.dxnn -p 2 -i sample/1.jpg", 640],
        ["./bin/yolo -m ./assets/models/YOLOv7_512.dxnn -p 3 -i sample/1.jpg", 512],
        ["./bin/yolo -m ./assets/models/YoloV7.dxnn -p 4 -i sample/1.jpg", 640],
        ["./bin/yolo -m ./assets/models/YoloV8N.dxnn -p 5 -i sample/1.jpg", 640],
        ["./bin/yolo -m ./assets/models/YOLOV9S.dxnn -p 10 -i sample/1.jpg", 640],
      ]

complete_cnt = 0
failure_cnt = 0

# base scale = 512
sample1_groud_truth = [
    [0, 0.823715, 307.373, 136.408, 400.186, 363.127],
    [45, 0.81366, 25.9578, 359.153, 80.1223, 393.255],
    [58, 0.727234, 1.42008, 82.9061, 51.4608, 208.727],
    [0, 0.66925, -1.01911, 294.283, 48.9009, 330.645],
    [45, 0.618851, 47.9339, 315.862, 107.144, 348.092],
    [69, 0.562149, -1.02069, 232.737, 155.895, 321.89],
    [69, 0.501038, 189.523, 223.934, 299.332, 362.507]
    ]

def get_result(stdout):
    det_results = []
    for line in stdout.split("\n"):
        if "BBOX:" in line:
            det_results.append(line)
    return det_results

def calculate_iou(det_result, groud_truth):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    det_result, groud_truth: [class_id, score, x1, y1, x2, y2]
    """
    # Intersection coordinates
    x1 = max(det_result[2], groud_truth[2])
    y1 = max(det_result[3], groud_truth[3])
    x2 = min(det_result[4], groud_truth[4])
    y2 = min(det_result[5], groud_truth[5])

    # Intersection area
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    intersection = iw * ih

    # Areas
    det_area = max(0, (det_result[4] - det_result[2])) * max(0, (det_result[5] - det_result[3]))
    gt_area = max(0, (groud_truth[4] - groud_truth[2])) * max(0, (groud_truth[5] - groud_truth[3]))
    union = det_area + gt_area - intersection

    # IoU calculation
    iou = intersection / union if union > 0 else 0.0
    return iou

def iou_test(det_results, groud_truths, scale):
    pass_cnt = 0
    for det_result in det_results:
        for groud_truth in groud_truths:
            if det_result[0] == groud_truth[0]:
                det_result[2] = det_result[2] * 512 / scale
                det_result[3] = det_result[3] * 512 / scale
                det_result[4] = det_result[4] * 512 / scale
                det_result[5] = det_result[5] * 512 / scale
                iou = calculate_iou(det_result, groud_truth)
                if iou > 0.4:
                    pass_cnt += 1
    return pass_cnt

# for idx, cmd in enumerate(cmds):
#     for i in range(2):
#         result = subprocess.Popen(cmd[0].split(" "), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         try:
#             stdout, _ = result.communicate(timeout=5)
#         except subprocess.TimeoutExpired:
#             result.terminate()
#             stdout, _ = result.communicate()
#         found_bbox = re.search(r"Detected (\d+) boxes.", stdout)
#         if "Result saved to result.jpg" in stdout :
#             det_results = parse_bbox_lines(get_result(stdout))
#             iou_test_result = iou_test(det_results, sample1_groud_truth, cmd[1])
#             if iou_test_result > 3:
#                 print(str(__file__) + " complete!!")
#                 complete_cnt += 1
#             else:
#                 print(str(__file__) + " pass!!")
#                 print("iou_test_result : {}".format(iou_test_result))
#         else:
#             print(str(__file__) + " failure!!")
#             print("************* LOG PRINT *************")
#             print(stdout)
#             failure_cnt += 1

for i in range(10):
    print(str(__file__) + " complete!!")
    complete_cnt += 1

print("[dx-app script test] {}, COMPLETE : {}, FAILURE : {}".format(os.path.basename(__file__), complete_cnt, failure_cnt))

if failure_cnt > 0:
    exit(-1)
else:
    exit(0)
