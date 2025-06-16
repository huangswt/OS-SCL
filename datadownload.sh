#!/bin/bash

# Create directory
mkdir -p data/dataset
mkdir -p data/eval_dataset

# Download and decompress development datasets
cd data/dataset
wget -O dev_data_fan.zip https://zenodo.org/records/3678171/files/dev_data_fan.zip?download=1 &
wget -O dev_data_pump.zip https://zenodo.org/records/3678171/files/dev_data_pump.zip?download=1 &
wget -O dev_data_slider.zip https://zenodo.org/records/3678171/files/dev_data_slider.zip?download=1 &
wget -O dev_data_ToyCar.zip https://zenodo.org/records/3678171/files/dev_data_ToyCar.zip?download=1 &
wget -O dev_data_ToyConveyor.zip https://zenodo.org/records/3678171/files/dev_data_ToyConveyor.zip?download=1 &
wget -O dev_data_valve.zip https://zenodo.org/records/3678171/files/dev_data_valve.zip?download=1 &

# Wait for all download tasks to complete
wait

# Extract the downloaded data set
unzip dev_data_fan.zip
unzip dev_data_pump.zip
unzip dev_data_slider.zip
unzip dev_data_ToyCar.zip
unzip dev_data_ToyConveyor.zip
unzip dev_data_valve.zip

rm *.zip



cd ..
cd eval_dataset

# Download the evaluation dataset in parallel

wget -O eval_data_train_fan.zip https://zenodo.org/records/3727685/files/eval_data_train_fan.zip?download=1 &
wget -O eval_data_train_pump.zip https://zenodo.org/records/3727685/files/eval_data_train_pump.zip?download=1 &
wget -O eval_data_train_slider.zip https://zenodo.org/records/3727685/files/eval_data_train_slider.zip?download=1 &
wget -O eval_data_train_ToyCar.zip https://zenodo.org/records/3727685/files/eval_data_train_ToyCar.zip?download=1 &
wget -O eval_data_train_ToyConveyor.zip https://zenodo.org/records/3727685/files/eval_data_train_ToyConveyor.zip?download=1 &
wget -O eval_data_train_valve.zip https://zenodo.org/records/3727685/files/eval_data_train_valve.zip?download=1 &


wget -O eval_data_test_fan.zip https://zenodo.org/records/3841772/files/eval_data_test_fan.zip?download=1 &
wget -O eval_data_test_pump.zip https://zenodo.org/records/3841772/files/eval_data_test_pump.zip?download=1 &
wget -O eval_eval_data_test_slider.zip https://zenodo.org/records/3841772/files/eval_data_test_slider.zip?download=1 &
wget -O eval_data_test_ToyCar.zip https://zenodo.org/records/3841772/files/eval_data_test_ToyCar.zip?download=1 &
wget -O eval_data_test_ToyConveyor.zip https://zenodo.org/records/3841772/files/eval_data_test_ToyConveyor.zip?download=1 &
wget -O eval_data_test_valve.zip https://zenodo.org/records/3841772/files/eval_data_test_valve.zip?download=1 &





# Wait for all download tasks to complete
wait

# Extract the downloaded evaluation data set
unzip eval_data_train_fan.zip
unzip eval_data_train_pump.zip
unzip eval_data_train_slider.zip
unzip eval_data_train_ToyCar.zip
unzip eval_data_train_ToyConveyor.zip
unzip eval_data_train_valve.zip

#Decompressed evaluation set
unzip eval_data_test_fan.zip
unzip eval_data_test_pump.zip
unzip eval_eval_data_test_slider.zip
unzip eval_data_test_ToyCar.zip
unzip eval_data_test_ToyConveyor.zip
unzip eval_data_test_valve.zip

rm *.zip






cd fan
mv train/* ../../dataset/fan/train/
cd ..

cd pump
mv train/* ../../dataset/pump/train/
cd ..

cd slider
mv train/* ../../dataset/slider/train/
cd ..

cd ToyCar
mv train/* ../../dataset/ToyCar/train/
cd ..

cd ToyConveyor
mv train/* ../../dataset/ToyConveyor/train/
cd ..

cd valve
mv train/* ../../dataset/valve/train/
cd ..



