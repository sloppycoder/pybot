
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
# OpenAI API currently runs very slow due to high demand, each row requires 2-3 seconds
# to execute, for now this test only uses first 1500 rows
# this test case will create file data/test1.csv which will be used in the next step
# setting DEBUG=1 will display result payload from openai APIs
# redis cache is used to store openai API response to reduce debug time.
# update classifier.py if redis is not running at localhost:3679 without authentication
# to delete cache entries from this program, set CLEAR_CACHE=1 when running the following
#
# this step can take MANY HOURS to run

DEBUG=1 pytest -s -k test_extract_features


# read data/test1.csv from the previous test and feed into XGBoost for model training
# the code is a VERY ROUGH PROTOTYPE and should be further tuned before serious use
# with limited data of 1k entires the accuracy is 70%

pytest -s -k test_classify

```

## TODOs

1. re-run feature extract for all 5400 items (will take some time to run)
2. refactor and tune classification model
