# Gait Cleaning Pipeline

The original parent paper for the gait analysis did not come with published code for their data cleaning. Since I will be working with their final cleaned data, I must execute a version of their code to satisfy project requirements. 

## Methods

A cleaning pipeline was reverse-engineered by using the available raw and clean datasets for each subject. The pipeline fully runs without the existing clean data, and generates a RECONSTRUCTED/; a folder that contains the closest approximation of the cleaned data. A copy of the cleaned data can be provided alongside the raw data - the pipleline will use this to verify the reconstruction against the actual data and generate an accuracy report. In the absence of the original cleaned data, this report will be skipped. 

## Workflow

The following dependencies are required to run this pipeline: 

```bash
pip install numpy pandas scipy
```

The pipeline should be run with the original RAW and optional CLEANED data folders in the same root directory. No separate directories are necessary. 

## Data

The original datasets can be found at these links: 

RAW: [web download](https://figshare.com/articles/dataset/Raw_Data/30399988?backTo=/collections/The_cognitive-motor_coordination_dataset_gait_analysis_under_combined_blindfold_virtual_reality-induced_cognitive_load_and_vibrotactile_feedback/8584895)

CLEANED: [web download](https://figshare.com/articles/dataset/Cleaned_Data/32948690?backTo=/collections/_/8584895)
