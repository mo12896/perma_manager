# To run this script, you need a config.json file in the same folder as the script.
# It should follow this template:
# {
#    "teams": [
#        {
#            "name": "team_xx",
#            "days": [
#                {
#                    "date": "Year-Month-Day",
#                    "clips": [
#                        {
#                            "input_folder": path to the folder where the video is stored,
#                            "input_video": name of the video,
#                            "output_folder": path to the folder where the video will be stored,
#                            "start_times": list of start times in seconds,
#                            "end_times": list of end times in seconds
#                        },
#                    ],
#                }
#            ]
#        }
#    ]
# }
#

from pathlib import Path
from moviepy.editor import VideoFileClip
import json
from threading import Thread
from time import perf_counter


def cut_video(
    input_video: Path,
    start_times: int,
    end_times: int,
    out_folder: Path,
    video_idx: int,
):
    """Function to cut a video into multiple clips.

    Args:
        input_video (Path): The path to the video to be cut.
        start_times (int): Start time of the clip in seconds.
        end_times (int): End time of the clip in seconds.
        out_folder (Path): The path to the folder where the clips will be saved.
        idx (int): The index of the video, the clip is created from.
    """

    out_folder.mkdir(parents=True, exist_ok=True)
    for start, end in zip(start_times, end_times):
        try:
            clip = VideoFileClip(str(input_video)).subclip(start, end)
            clip_path = out_folder / f"clip_{video_idx}_{start}_{end}.mp4"
            if not clip_path.exists():
                #clip.write_videofile(str(clip_path))
                clip_25fps = clip.set_fps(25)
                clip_25fps.write_videofile(str(clip_path))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    with open("config.json") as json_file:
        data = json.load(json_file)
        threads = []

        for team in data["teams"]:
            for day in team["days"]:
                t1_start = perf_counter()

                for video_idx, clip in enumerate(day["clips"]):
                    input_folder = Path(clip["input_folder"])
                    if not input_folder.exists():
                        raise ValueError("Input folder does not exist.")
                    input_video = input_folder / clip["input_video"]
                    if not input_video.exists():
                        raise ValueError("Input video does not exist.")
                    start_times = clip["start_times"]
                    end_times = clip["end_times"]
                    if len(start_times) != len(end_times):
                        raise ValueError("Number of start and end times do not match.")
                    for start, end in zip(start_times, end_times):
                        if start >= end:
                            raise ValueError("Start time is greater than end time.")
                    output_folder = Path(clip["output_folder"])
                    if not output_folder.exists():
                        raise ValueError("Output folder does not exist.")
                    team_name = team["name"]
                    date = day["date"]
                    out_folder = output_folder / team_name / date

                    t = Thread(
                        target=cut_video,
                        args=(
                            input_video,
                            start_times,
                            end_times,
                            out_folder,
                            video_idx,
                        ),
                    )
                    threads.append(t)
                    t.start()

                t1_stop = perf_counter()
                print(f"Elapsed time: {round((t1_stop - t1_start) / 60, 2)} minutes!")

        for thread in threads:
            thread.join()
