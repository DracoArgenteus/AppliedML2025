# Installation for MacOS

First make sure you have anaconda installed, which you can find [here](https://www.anaconda.com/download/). Open the `.pkg` file and install it.

I recommend creating the virtual environment with conda. The instructions that follow should work regardless of operating system (i.e. also on Linux/Windows).

```conda create -n app-ml python==3.12.10```

Note that `3.13` wasn't working well with package versions for some people.

One difficulty in solving for the packages were version conflicts and availability. The most recent version of tensorflow on its conda repository is 2.18. I modified the tensorflow version in `requirements.txt` and set it to `tensorflow>=2.18.0`. Also, I commented out the installation for the `torch` packages also due to conflicts (which can later be installed via pip).

The good thing is, one can use the file with conda as well

To do that, activate the environment we just created:

```conda activate app-ml```

and install the packages in the .txt file as

```conda install -c conda-forge --file requirements.txt```

After having done this, you can install the `torch` family packages via `pip`

```pip install torch==2.6.0 torch-geometric==2.6.1 torchaudio==2.6.0 torchvision==0.21.0```

and you should be good to go.

Type `jupyter lab`, which will open Jupyter in a new browser tab/window. Then try to run the code in the `Week0` and `Week1` folders. 

Remember, when you restart your computer you will have to activate the `app-ml` environment again! (`conda activate app-ml`)
### Exceptional Cases

A few people got a long `Kernel Error` while trying to run the notebooks in VSCode or even a local `jupyter` server, which was somewhere complaining about `_sqlite` or libsqlite not being found.

If (and only if) you're getting that error, then try

```conda install --force-reinstall ipykernel libsqlite```

