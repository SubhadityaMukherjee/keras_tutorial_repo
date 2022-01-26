#%%
import os
from glob import glob
from pathlib import Path

#%%
main_str = """
# Keras Tutorials

- Just a repo with codes for various tasks using Keras/Tensorflow.
- Most of them so far are from [keras](keras.io) but more will be added as and when I learn about tasks myself. 
- Many codes are not true to the original repos.
- This is by no means comprehensive, but should be easy enough to take the codes for a specific task and get a concept working ASAP.

## A huge thank you to

- [keras.io](https://keras.io/examples/)

## FAQ
- Why are there notebook and py files for the same thing? Well. I prefer running things using py files and VSCode's IPython extension. Many others love notebooks. Each to their own. 
All the notebooks are auto generated and so have no inputs

## INDEX

"""
subdirs = glob("./*/", recursive=True)
subdirs = [x for x in subdirs if not any(["Drafts" in x])]
subdirs


def return_git_link(x, name=None):
    x = x.strip()[1::]
    if name == None:
        x = f"  - [{x.split('/')[-1]}](https://github.com/SubhadityaMukherjee/keras_tutorial_repo/tree/main{x})\n"
    else:
        x = f"- [{name[2::]}](https://github.com/SubhadityaMukherjee/keras_tutorial_repo/tree/main{x})\n"
    return x


for folder in subdirs:

    main_str = main_str + return_git_link(folder, folder) + "\n"
    every_folder = [return_git_link(x) for x in glob(f"{folder}*", recursive=True)]
    main_str = main_str + "\n".join(every_folder) + "\n"

with open("README.md", "w+") as f:
    f.write(main_str)
