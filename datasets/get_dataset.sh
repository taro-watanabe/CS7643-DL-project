#!/bin/bash

set -e

# Variables
force_refresh=false
data1_url="https://github.com/NetManAIOps/KPI-Anomaly-Detection/raw/master/Finals_dataset/phase2_ground_truth.hdf.zip"
data1_file="data1.zip"
data_new_url="https://github.com/NetManAIOps/KPI-Anomaly-Detection/raw/master/Finals_dataset/phase2_train.csv.zip"
data_new_file="datanew.zip"
intrusion_data_url="http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"
intrusion_data_file="intrusion_data.zip"
nab_path="https://github.com/numenta/NAB"

# Check for --force-refresh option
for arg in "$@"
do
  case $arg in
    --force-refresh)
    force_refresh=true
    shift
    ;;
  esac
done

# Download files
download_file() {
  url=$1
  file=$2

  if [ ! -f "$file" ] || [ "$force_refresh" = true ]; then
    echo "Downloading $file..."
    wget -O "$file" "$url"
  else
    echo "$file already exists. Skipping download."
  fi
}

download_file "$data1_url" "$data1_file"
download_file "$data_new_url" "$data_new_file"
download_file "$intrusion_data_url" "$intrusion_data_file"

# Unpack files
unzip_files() {
  file=$1
  dir=$2

  echo "Unzipping $file to $dir..."
  mkdir -p "$dir"
  unzip -o "$file" -d "$dir"
}

unzip_files "$data_new_file" "univariate_datasets/"
unzip_files "$intrusion_data_file" "intrusion_datasets/"

# Find CSV files and create folders
find_csv_files() {
  src_dir=$1
  dest_dir=$2

  echo "Finding CSV files in $src_dir and moving them to $dest_dir..."
  mkdir -p "$dest_dir"
  find "$src_dir" -iname "*.csv" -exec mv {} "$dest_dir/" \;
  find "$src_dir" -type d -empty -delete
}

find_csv_files "univariate_datasets/" "csv_datasets/univariate/"
find_csv_files "intrusion_datasets/" "csv_datasets/intrusion/"

# skip if NAB folder already exists
if [ -d "NAB" ]; then
  echo "NAB folder already exists. Skipping download."
else
  echo "Downloading NAB dataset..."
  git clone ${nab_path}
fi
python3 preprocess_nab.py


# Open Anomaly Detection Datasets Taken from benchmarking

# skip if open-anomaly-detection-benchmark folder already exists
if [ -d "open-anomaly-detection-benchmark" ]; then
  echo "open-anomaly-detection-benchmark folder already exists. Skipping download."
else
  echo "Downloading open-anomaly-detection-benchmark dataset..."
  git clone git@github.com:shahzaib-ch/open-anomaly-detection-benchmark.git 

fi

# copy to csv_datasets if not already there
if [ -d "csv_datasets/open-anomaly-detection-benchmark" ]; then
  echo "open-anomaly-detection-benchmark folder already exists in csv_datasets. Skipping copy."
else
  echo "Copying open-anomaly-detection-benchmark dataset to csv_datasets..."
  cp -r open-anomaly-detection-benchmark/data/ csv_datasets/open-anomaly-detection-benchmark
fi
