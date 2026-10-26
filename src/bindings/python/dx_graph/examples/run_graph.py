#!/usr/bin/env python3
"""Run a multi-model graph JSON from Python with dx_graph - the Python
counterpart of `multi_model_graph_async --graph ... --output ...`.

    PYTHONPATH=bin/python python3 src/bindings/python/dx_graph/examples/run_graph.py \
        --graph src/cpp_example/multi_model_graph/fanout_od_seg_depth.json \
        --output-dir /tmp/renders

Streams the graph's own source (or --input), prints one line per frame -
a failed frame's line on stderr, as the CLI prints it - and writes each
frame's rendering: PNG through OpenCV when the cv2 module is installed,
otherwise a binary PPM written with numpy alone. Exits 1 when any frame
failed, like the CLI. A graph with several sources reads them all, one
stream each, and names the stream in every line (`stream cam1 frame 0: ...`)
and file name (`frame_cam1_000000.png`).
"""
import argparse
import os
import sys

import dx_graph


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run a dx_app multi-model graph from Python (dx_graph).")
    parser.add_argument("--graph", required=True, help="graph JSON file")
    parser.add_argument("--model-dir", default=None,
                        help="directory of .dxnn files (default: <repo>/assets/models)")
    parser.add_argument("--input", default=None,
                        help="image or video replacing the graph's only source node's uri, "
                             "as the CLI's --input")
    parser.add_argument("--executor", choices=("sync", "async"), default="async")
    parser.add_argument("--max-inflight", type=int, default=16,
                        help="frames in flight (async only), as the CLI's --max-inflight")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after this many frames; 0 = all, as the CLI's --frames")
    parser.add_argument("--output-dir", default=None,
                        help="write each frame's rendering here (created if absent)")
    return parser.parse_args(argv)


def image_writer():
    """(extension, write(path, bgr)) - OpenCV when importable, else PPM."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        def write(path, canvas):
            if not cv2.imwrite(path, canvas):
                raise IOError("could not write " + path)
        return ".png", write

    def write_ppm(path, canvas):
        rows, cols = canvas.shape[:2]
        with open(path, "wb") as handle:
            handle.write("P6\n{} {}\n255\n".format(cols, rows).encode("ascii"))
            handle.write(canvas[:, :, ::-1].tobytes())  # PPM is RGB
    return ".ppm", write_ppm


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        extension, write = image_writer()

    try:
        graph = dx_graph.Graph(args.graph, model_dir=args.model_dir,
                               executor=args.executor,
                               max_frames_in_flight=args.max_inflight)
    except dx_graph.GraphError as error:
        print(error, file=sys.stderr)  # the CLI's own message
        return 1

    frames = failed = 0
    with graph:
        several = len(graph.streams) > 1
        try:
            reports = graph.stream(source=args.input, limit=args.frames)
        except ValueError as error:  # e.g. --input on a graph with several sources
            print(error, file=sys.stderr)
            return 1
        try:
            for report in reports:
                frames += 1
                label = ("stream {} frame {}".format(report.stream, report.frame_index)
                         if several else "frame {}".format(report.frame_index))
                if report.error:
                    failed += 1
                    print("{}: {}".format(label, report.error),
                          file=sys.stderr)  # as the CLI prints a failed frame
                else:
                    d = report.to_dict(arrays=False)
                    nodes = len(d["nodes"]) + len(d.get("roi_nodes", {}))
                    print("{}: {} node{}".format(label, nodes, "" if nodes == 1 else "s"))
                if args.output_dir:
                    name = ("frame_{}_{:06d}{}".format(report.stream, report.frame_index, extension)
                            if several else "frame_{:06d}{}".format(report.frame_index, extension))
                    write(os.path.join(args.output_dir, name), graph.render(report))
        except dx_graph.GraphError as error:  # e.g. an input that will not open
            print(error, file=sys.stderr)
            return 1

    print("{} frame{}, {} failed".format(frames, "" if frames == 1 else "s", failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
