import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta

def plot(input, output, start_ratio, end_ratio, show_gap):
    print("Input : ", input)
    print("Output : ", output)
    # Read profiler json file
    with open(input, "r") as json_file:
        json_data = json.load(json_file)

    # Define entire interval
    all_timings = []
    for event_timing_list in json_data.values():
        all_timings.extend(event_timing_list)
    min_start = min(event_timing["start"] for event_timing in all_timings)
    max_end = max(event_timing["end"] for event_timing in all_timings)
    interval_all = max_end - min_start

    # Calculate actual time window
    interval_start = min_start + interval_all*start_ratio
    interval_end = min_start + interval_all*end_ratio
    interval = interval_end - interval_start

    # Setup plot
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    # Visualize by event
    colors = list(mcolors.TABLEAU_COLORS.values())
    multi = int(len(json_data.keys())/len(colors) + 1)
    colors *= multi
    last_start = 0
    for idx, (event_name, timing_data) in enumerate(json_data.items()):
        sorted_timings = sorted(timing_data, key=lambda x: x["start"])  # Sort by start time
        # sorted_timings = timing_data
        for i, event_timing in enumerate(sorted_timings):  # Loop through all timings
            start = event_timing["start"]
            end = event_timing["end"]
            # Check if the timing is within the specified time window
            if start > 0 and end > 0 and start >= interval_start and end <=interval_end and end>start :
                start = start - interval_start
                end = end - interval_start
                duration = end - start
                color = colors[idx]
                rect = plt.Rectangle((start, idx), duration, 0.7, linewidth=1, edgecolor='g', facecolor=color, alpha=0.5)
                ax.add_patch(rect)
                text_x = start + duration / 2
                text_y = idx + 0.3
                text_fontsize = 5
                if i % 2 == 0:  # even
                    plt.text(text_x, text_y - 0.15, f"{duration/1000}us", ha='center', va='center', color='black', fontsize=text_fontsize)
                else:  # odd
                    plt.text(text_x, text_y + 0.15, f"{duration/1000}us", ha='center', va='center', color='black', fontsize=text_fontsize)
                # Calculate and display the time difference between current and last starting point
                if show_gap:
                    if i>0 and start > last_start:
                        time_difference = start - last_start
                        plt.text((last_start+start)/2, text_y - 0.4, f"Δ {time_difference/1000}us", ha='center', va='center', color='black', fontsize=text_fontsize)
                    last_start = start
                # print("pick: ", interval_start, event_name, event_timing, duration) # debug
            else:
                duration = end - start
                # print("drop: ", interval_start, event_name, event_timing, duration) # debug

    plt.xlabel("Time")
    plt.ylabel("Event")
    plt.title("DX-RT Profiler")
    plt.yticks(range(len(json_data)), json_data.keys())
    plt.ylim(-0.1, len(json_data))
    plt.xlim(0, interval )
    plt.savefig(output, bbox_inches="tight", dpi=300)
    print(interval,"ns / ", interval_all, "ns")
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw timing chart from profiler data.")
    parser.add_argument("-i", "--input", default="profiler.json", help="Input json file to plot")
    parser.add_argument("-o", "--output", default="profiler.png", help="Output image file to save the plot")
    parser.add_argument("-s", "--start", type=float, default=0.0, help="Starting point( > 0.0) when the entire interval is 1")
    parser.add_argument("-e", "--end", type=float, default=1.0, help="End point( < 1.0) when the entire interval is 1")
    parser.add_argument("-g", "--show_gap", action="store_true", help="Show time gap between starting points")
    args = parser.parse_args()
    input = args.input
    output = args.output
    start_ratio = args.start
    end_ratio = args.end
    show_gap = args.show_gap
    
    plot(input, output, start_ratio, end_ratio, show_gap)