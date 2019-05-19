#!/bin/bash

function move_files {
	json_list=$(ls $1 | grep -E "*.json" | shuf -n 5)

	for json in $json_list; do
		file=${json%.*}

		center_camera=$file"-0.mov"
		left_camera=$file"-1.mov"
		right_camera=$file"-2.mov"

		mv $src"/"$json $2
		mv $src"/"$center_camera $2
		mv $src"/"$left_camera $2
		mv $src"/"$right_camera $2
	done
}


src="train_raw"
dst1="validation_raw"
dst2="test_raw"

if ! [ -d $dst1 ]; then
	mkdir $dst1
fi

if ! [ -d $dst2 ]; then
	mkdir $dst2
fi

move_files "$src" "$dst1"
move_files "$src" "$dst2"