
# Test project to use LLM APIs for various tasks

This project contains prototype code for using OpenAI and Google LLM for various tasks.

1. Use OpenAI for chat completion
2. Use OpenAI chat completion for code generation
3. Use OpenAI chat completion to extract features from simple text strings and use the features to train XGBoost classification model

## Setup

The easiest way to get started is probably use [Jetpack.io devbox](https://www.jetpack.io/devbox). Install devbox first, then

```shell
devbox shell

# you should ready to go

```

The more traditional way is to install python 3.11 and [poetry](https://python-poetry.org/), then

```shell

# create virtualenv
poetry shell
# install dependencies
poetry install

```

## 1. Use OpenAI for chat completion

```shell
pytest -s -k test_openai_completion

```

## 2. Use OpenAI chat completion for code generation

```shell
pytest -s -k test_openai_codegen

```

## 3. Use OpenAI chat completion to extract features from simple text strings

These tests requires some propierty data files that are not in the project repository.

```shell
# get the input file set1.csv
mkdir data
cp <source_file> data/set1.csv

# run feature extract to call OpenAI API to extract features
# the result will be saved in result.pickle file in Python pickle format
# OpenAI API currently runs very slow due to high demand, each row requires 2-3 seconds
# to execute, for now this test only uses first 1000 rows
pytest -s -k test_extract_with_openai

# read data/result.pick file from the previous test and convert into data/1k.csv
# file which can be used to train XGBoost model in next step
# this step does not require internet access
cp result.pickle data/result_1k.pickle
pytest -s -k test_data_prep

# read data/1k.csv file and train model
# DISCLAIMER: the code is a VERY ROUGH PROTOTYPE and should be further tuned before serious use
# with limited data of 1k entires the accuracy is 70%

pytest -s -k test_classify

```

## TODOs

1. re-run feature extract for all 5400 items (will take some time to run)
2. refactor and tune classification model
