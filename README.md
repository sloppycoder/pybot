
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

# create a file .env with following entry
OPENAI_API_KEY=sk-xxxxx

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
# get the input file test1.xlsx
mkdir data
cp <source_file> data/test1.xlsx

# run feature extract to call OpenAI API to extract features
# OpenAI API to extract features from a given part description.
# at the moment (Nov 2023), gpt-3.5-turbo-1106 seems to have simliar output to gpt-4
# and runs much faster (and cheaper too).
#
# this test case will create file data/test1.csv which will be used in the next step
# setting DEBUG=1 will display result payload from openai APIs
#
# the features extract from openai API will be saved in cache directory
# so re-running this test will not trigger API calls unless the cache is deleted
#

pytest --log-cli-level=DEBUG -s -k test_extract_features


# read data/test1.csv from the previous test and feed into XGBoost for model training
# the code is a VERY ROUGH PROTOTYPE and should be further tuned before serious use
# currently the accuray is 75% using the full dataset of 5300 records
#
# this step will save the model and encoders
#  data/feature1_model.joblib
#  data/feature1_combined_features.joblib
#  data/feature1_encoder.joblib

pytest -s -k test_classify

# load model from disk, lo and run one prediction
pytest -s --fromfile=tests/parts.txt -k test_batch_predict

# read all rows from data/test1.csv and run prediction using the trained model
# the result will be saved to data/compare1.csv.
# currently the hit ratio is 86%
# need new data to verify
pytest -s -k classify


```

## TODOs

1. re-run feature extract for all 5400 items (will take some time to run) (done)
2. improve feature extraction to more accurately extract relevant features
3. tune the model training logic
