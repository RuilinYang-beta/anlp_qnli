# what we have now

## these are probably mature enough to be used by others

- utility functions in `utils.py` about processing data, see the doc in the file
  - Note that `xy_pair_generator` might subject to change - right now it yields one pair of (x,y) at a time, but we might want it to be better, such as randomly shuffle the order of the training examples per epoch, and yielding m pairs of (x,y) at a time for mini-batch training.
    - or we keep it as it is (so that things depend on it won't break) and later get a new thing to deal with loading data in batch?
- `statics.py` stores the constants.

## they are immature but they are a demo of training loop, among others

- `temp_playground.ipynb` is for playing with things that look weird to help understanding, maybe what's weird to me is also weird to you? feel free to add your weird things
- `temp_playground.py` is a demo training loop of `SimpleRNN`
- `SimpleRNN.py` is almost finished, it can be better if we figure out how to do batch training

# what to do next?

- for BiRNN: I don't know, comrade Ellie you define what you want to do; perhaps play around in a standalone file and make sure it works? because the `qnli.py` looks scary 🙃 we can put the cleaned-up final version in that file
- for SimpleRNN: figure out how to do batch training
- for stacked RNN: how to do it
- general purpose:
  - [high] evaluation function
  - [mid-high] data loader and batch training
  - [mid] error analysis + hyperparam tuning
  - (optional) some plotting function
