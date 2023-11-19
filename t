  #! /bin/bash

  BATCH=$1

  if [ "$BATCH" = "" ]; then
    BATCH=tmp
  fi

  pytest -s -k test_extract_features --batch $BATCH
  pytest -s -k test_train_model_with_feature --batch $BATCH
  pytest -s -k test_batch_predict --batch $BATCH
  pytest -s -k test_batch_predict --batch $BATCH
  pytest -s -k test_predict_all --batch $BATCH
