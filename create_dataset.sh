#!/bin/bash

in_train="train_raw"
out_train="train"

in_validation="validation_raw"
out_validation="validation"

in_test="test_raw"
out_test="test"

# if ! [ -d $out_train ]; then
# 	mkdir $out_train
# fi

# python3 create_dataset.py --src $in_train --dst $out_train


# if ! [ -d $out_validation ]; then
# 	mkdir $out_validation
# fi

# python3 create_dataset.py --src $in_validation --dst $out_validation

if ! [ -d $out_test ]; then
	mkdir $out_test
fi

python3 create_dataset.py --src $in_test --dst $out_test