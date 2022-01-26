#!/bin/bash
black "." && isort .
# create scripts if there are any notebook files
for i in $(fdfind --glob "*.py"); 
do 
		jupytext --to notebook $i; 
done
python indexer.py
rm indexer.ipynb
if [ ! -z $1 ]; then
        git add . && git commit -m $1 && git push
fi
