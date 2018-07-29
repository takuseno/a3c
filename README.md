# A3C
A3C imeplementation with Distributed TensorFlow.

Multiple processes instead of multiple threads execute the algorithm as GIL is critical to run many simulators.

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
$ ./run.sh [-e env_id (environment name)] [-n num_of_process (the number of processes)] [-r (render)]
```

### TensorBoard
```
$ tensorboard --logdir logs
```
