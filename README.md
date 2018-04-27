# A3C
A3C imeplementation with TensorFlow.

## requirements
- Python3

## dependencies
- tensorflow
- opencv-python
- numpy
- git+https://github.com/imai-laboratory/rlsaber

## implementations
This repostory is inspired by following projects.

- [OpenAI Baselines](https://github.com/openai/baselines)
- https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

## train
```
$ python train.py [--env environment name] [--threads thread number] [--render] [--demo]
```

### TensorBoard
```
$ tensorboard --logdir logs
```
