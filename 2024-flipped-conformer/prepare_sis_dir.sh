#!/bin/bash

set -e

dirname=$1

if [ ! $# -eq 1 ]; then
  echo "Usage: $0 <setup-dirname>"
  exit -1
fi

mkdir -p $dirname
cd $dirname

echo "Cloning repos"

git clone git@github.com:rwth-i6/sisyphus.git
ln -s sisyphus/sis .  # used to call the manager to run jobs

git clone git@github.com:rwth-i6/returnn.git

mkdir "recipe"
git clone git@github.com:rwth-i6/i6_core.git recipe/i6_core
git clone git@github.com:rwth-i6/i6_experiments.git recipe/i6_experiments

echo "Creating setup dirs"

mkdir "alias"  # sis jobs aliases (symlinks)
mkdir "output"  # sis jobs outputs (symlinks)
mkdir "work"  # sis jobs
mkdir "config"  # main entry config is added here

# sisyphus settings which needs to be adapted depending on the working environment
cp recipe/i6_experiments/users/schmitt/experiments/exp2024_08_27_flipped_conformer/settings.py .

# create dir for pycharm project to avoid indexing huge dirs (e.g work)
echo "Creating dir for PyCharm project"

pycharmProj="`basename $PWD`-proj"
mkdir $pycharmProj
cd $pycharmProj
ln -s ../returnn .
ln -s ../sisyphus .
ln -s ../settings.py .
ln -s ../recipe .
ln -s ../config .
