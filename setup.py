import os

# This file creates all folder necessary for the project to store the results
result_folder_names = ["ShearWaveDecayResults", "CouetteFlowResults", "PoiseuilleFlowResults", "SlidingLidResults"]

# create the results folders
for folder_name in result_folder_names:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)