# boids
boid simulation using python and pygame

requires numpy and pygame

If you have Anaconda installed you can run by doing the following:

```
conda create -n boids python=3.8 numpy
conda activate boids
pip install pygame
git clone https://github.com/AndrewBohler/boids
cd boids
python boids.py
```

pygame is not hosted on Anaconda so you have to install it with pip

If you are on Mac I suggest using pyenv and then:

```
git clone https://github.com/AndrewBohler/boids
cd boids
pyenv install $(cat .python-version)
pyenv activate
pip install -r requirements.txt
python boids.py
```

If you are using zsh and have the hook for pyenv enabled then the `.python-version` file should tell pyenv to use the right python version automatically when you cd into the `boids` folder

