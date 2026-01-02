def print_image_processing_summary(t_start, t0, t1, t2, t3, t4=None):
    read_time = (t0 - t_start) * 1000.0
    preprocess_time = (t1 - t0) * 1000.0
    inference_time = (t2 - t1) * 1000.0
    postprocess_time = (t3 - t2) * 1000.0

    print("\n" + "=" * 35)
    print(f" {'IMAGE PROCESSING SUMMARY':^35}")
    print("=" * 35)
    print(f" {'Pipeline Step':<15} {'Latency':>11}")
    print("-" * 35)
    print(f" {'Read':<15} {read_time:8.2f} ms")
    print(f" {'Preprocess':<15} {preprocess_time:8.2f} ms")
    print(f" {'Inference':<15} {inference_time:8.2f} ms")
    print(f" {'Postprocess':<15} {postprocess_time:8.2f} ms")

    if t4 is not None:
        draw_time = (t4 - t3) * 1000.0
        print(f" {'Display':<15} {draw_time:8.2f} ms")
        total_time = (t4 - t_start) * 1000.0
    else:
        total_time = (t3 - t_start) * 1000.0

    print("-" * 35)
    print(f" {'Total Time':<15} : {total_time:6.1f} ms")
    print("=" * 35)


def print_sync_performance_summary(
    metrics: dict, cnt: int, elapsed: float, display: bool
):
    """Print detailed performance metrics"""
    overall_fps = cnt / elapsed if elapsed > 0 else 0.0
    avg_read = metrics["sum_read"] / cnt * 1000.0
    avg_pre = metrics["sum_preprocess"] / cnt * 1000.0
    avg_inf = metrics["sum_inference"] / cnt * 1000.0
    avg_post = metrics["sum_postprocess"] / cnt * 1000.0
    read_fps = 1000.0 / avg_read if avg_read > 0 else 0.0
    pre_fps = 1000.0 / avg_pre if avg_pre > 0 else 0.0
    inf_fps = 1000.0 / avg_inf if avg_inf > 0 else 0.0
    post_fps = 1000.0 / avg_post if avg_post > 0 else 0.0
    print("\n" + "=" * 50)
    print(f"{'PERFORMANCE SUMMARY':^50}")
    print("=" * 50)
    print(f" {'Pipeline Step':<15} {'Avg Latency':<15} {'Throughput':<15}")
    print("-" * 50)
    print(f" {'Read':<15} {avg_read:8.2f} ms     {read_fps:6.1f} FPS")
    print(f" {'Preprocess':<15} {avg_pre:8.2f} ms     {pre_fps:6.1f} FPS")
    print(f" {'Inference':<15} {avg_inf:8.2f} ms     {inf_fps:6.1f} FPS")
    print(f" {'Postprocess':<15} {avg_post:8.2f} ms     {post_fps:6.1f} FPS")

    if display:
        avg_render = metrics["sum_render"] / cnt * 1000.0
        render_fps = 1000.0 / avg_render if avg_render > 0 else 0.0
        print(f" {'Display':<15} {avg_render:8.2f} ms     {render_fps:6.1f} FPS")

    print("-" * 50)
    print(f" {'Total Frames':<15} : {cnt:6d}")
    print(f" {'Total Time':<15} : {elapsed:6.1f} s")
    print(f" {'Overall FPS':<15} : {overall_fps:6.1f} FPS")
    print("=" * 50)


def print_async_performance_summary(
    metrics: dict, cnt: int, elapsed: float, display: bool
):
    """Print detailed performance metrics"""
    if metrics["infer_completed"] == 0:
        print("[WARNING] No frames were processed.")
        return

    overall_fps = cnt / elapsed if elapsed > 0 else 0.0
    avg_read = metrics["sum_read"] / metrics["infer_completed"] * 1000.0
    avg_pre = metrics["sum_preprocess"] / metrics["infer_completed"] * 1000.0
    avg_inf = metrics["sum_inference"] / metrics["infer_completed"] * 1000.0
    avg_post = metrics["sum_postprocess"] / metrics["infer_completed"] * 1000.0

    inflight_time_window = metrics["infer_last_ts"] - metrics["infer_first_ts"]
    infer_tp = (
        metrics["infer_completed"] / inflight_time_window
        if inflight_time_window > 0
        else 0.0
    )
    inflight_avg = (
        metrics["inflight_time_sum"] / inflight_time_window
        if inflight_time_window > 0
        else 0.0
    )
    inflight_max = metrics["inflight_max"]

    read_fps = 1000.0 / avg_read if avg_read > 0 else 0.0
    pre_fps = 1000.0 / avg_pre if avg_pre > 0 else 0.0
    post_fps = 1000.0 / avg_post if avg_post > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"{'PERFORMANCE SUMMARY':^50}")
    print("=" * 50)
    print(f" {'Pipeline Step':<15} {'Avg Latency':<15} {'Throughput':<15}")
    print("-" * 50)
    print(f" {'Read':<15} {avg_read:8.2f} ms     {read_fps:6.1f} FPS")
    print(f" {'Preprocess':<15} {avg_pre:8.2f} ms     {pre_fps:6.1f} FPS")
    print(f" {'Inference':<15} {avg_inf:8.2f} ms     {infer_tp:6.1f} FPS*")
    print(f" {'Postprocess':<15} {avg_post:8.2f} ms     {post_fps:6.1f} FPS")

    if display:
        avg_render = metrics["sum_render"] / metrics["infer_completed"] * 1000.0
        render_fps = 1000.0 / avg_render if avg_render > 0 else 0.0
        print(f" {'Display':<15} {avg_render:8.2f} ms     {render_fps:6.1f} FPS")

    print("-" * 50)
    print(" * Actual throughput via async inference")
    print("-" * 50)
    print(f" {'Infer Completed':<19} : {metrics['infer_completed']:6d}")
    print(f" {'Infer Inflight Avg':<19} : {inflight_avg:6.1f}")
    print(f" {'Infer Inflight Max':<19} : {inflight_max:6d}")
    print("-" * 50)
    print(f" {'Total Frames':<19} : {cnt:6d}")
    print(f" {'Total Time':<19} : {elapsed:6.1f} s")
    print(f" {'Overall FPS':<19} : {overall_fps:6.1f} FPS")
    print("=" * 50)
