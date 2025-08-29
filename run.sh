#!/bin/bash
docker build -t ai-finetuning-finance .
docker run --rm -it -v $(pwd):/app ai-finetuning-finance