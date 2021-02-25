"""Helpful script that allows for the visualization tool to be more easily used.
Usually, you would have to manually copy the track file in question over to the 
DR_USA_Intersection_MA folder, but this allows you to give the path of the track
file, and does the heavy lifting for you.
To use, simply run python3 sanitize_and_copy_csv.py "<path_to_csv>"
For example, python3 "dataset/DR_USA_Intersection_MA/train/tracks_000.csv"
"""
import os
import sys
import pandas as pd


def sanitize_and_copy_csv(csv_path):
    rootpath = os.path.join("..", ".")
    dir_listing = os.listdir(rootpath)
    if ("interaction-dataset" not in dir_listing):
        raise Exception("Please make sure that you have cloned \
            the dataset visualization tool into the project root directory.")
    if ("dataset" not in dir_listing):
        raise Exception("Please make sure that you have cloned \
            the Kaggle dataset into the project root directory.")

    last_slash_idx = csv_path.rfind("/")
    target_file = csv_path[last_slash_idx+1:]

    os.chdir(os.path.join(os.path.join("..", "."), csv_path[:last_slash_idx]))
    listing = os.listdir()

    if (target_file not in listing):
        raise Exception("File not found - please check the path and try again")

    df = pd.read_csv(target_file)
    df = df[pd.to_numeric(df['track_id'], errors='coerce').notnull()]
    column_titles = ["track_id", "frame_id", "timestamp_ms", "agent_type", "x", "y", "vx", "vy", "psi_rad", "length", "width"]

    df.drop(columns=['agent_role'])
    df = df.reindex(columns=column_titles)
    df['track_id'] = df['track_id'].astype(str)
    df = df.sort_values(by="track_id")

    # Now save the reformatted dataframe as a csv under interaction-dataset/recorded_trackfiles/DR_USA_Intersection_MA/vehicle_tracks_000.csv
    savetarget_dir = "interaction-dataset/recorded_trackfiles/DR_USA_Intersection_MA/"
    os.chdir("../../../")
    os.chdir(savetarget_dir)

    df.to_csv('vehicle_tracks_000.csv', index=False)