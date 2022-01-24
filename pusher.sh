#!/bin/bash
black "." && isort .
# create scripts if there are any notebook files
for i in $(fdfind --glob "*.py"); do jupytext --to notebook $i; done
if [ ! -z $1 ]; then
        git add . && git commit -m $1 && git push
fi
