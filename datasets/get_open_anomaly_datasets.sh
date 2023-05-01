#!/bin/bash

set -e
# skip if open-anomaly-detection-benchmark folder already exists
if [ -d "open-anomaly-detection-benchmark" ]; then
  echo "open-anomaly-detection-benchmark folder already exists. Skipping download."
else
  echo "Downloading open-anomaly-detection-benchmark dataset..."
  git clone https://github.com/shahzaib-ch/open-anomaly-detection-benchmark.git

fi

# copy to csv_datasets if not already there
if [ -d "csv_datasets/open-anomaly-detection-benchmark" ]; then
  echo "open-anomaly-detection-benchmark folder already exists in csv_datasets. Skipping copy."
else
  echo "Copying open-anomaly-detection-benchmark dataset to csv_datasets..."
  cp -r open-anomaly-detection-benchmark/data/ csv_datasets/open-anomaly-detection-benchmark
fi
