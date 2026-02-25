#!/bin/bash

if [ $# == 0 ]; then
	echo "Usage: $0 huggingface_model_id [target_folder]"
	echo "Example: $0 openai/gpt-oss-20b"
	exit
fi

if [ $# -ge 2 ]; then
	path="$2"
else
	path="`basename $1`"
fi

mkdir -p $path

if [ -s .secret.sh ]; then
	source .secret.sh
fi
HF_HUB_ENABLE_HF_TRANSFER=1 /opt/anaconda3/bin/hf download --local-dir ./$path --token $HF_TOKEN "$1" --exclude "original/*" "metal/*"

