"""Sanitizes and copies over the file, then runs the vis tool
To run this, do the following:
python3 run_vis.py "<path_to_csv_to_visualize>"
e.g. python3 run_vis.py "dataset/DR_USA_Intersection_MA/train/tracks_002.csv"
"""
import sanitize_and_copy_csv
import os
import sys

csv_path = sys.argv[1]
print(os.getcwd())
sanitize_and_copy_csv.sanitize_and_copy_csv(csv_path)
path_to_vis_tool = "interaction-dataset/python/"
vis_command = "python3 main_visualize_data.py DR_USA_Intersection_MA"
os.chdir("../../../")
print(os.getcwd())
os.chdir(path_to_vis_tool)
os.system(vis_command)