"""
Utility functions to : 
- prepare parser for command line arguments
- generate random number for hyperparameter sets
- write/add log to file
- load data from file
"""

import argparse
import json 
import random
import math
import os

def prepare_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("training",
                      choices=['normal', 'augmented'],
                      help="choose training set"
                      )  

  parser.add_argument("notation",
                      choices=['original', 'character'],
                      help="choose notation of number"
                      )

  parser.add_argument("model", 
                      choices=["FFNN", "RNN", "BiRNN", "Transformer"],
                      help="choose model to train"
                      )

  parser.add_argument("-e", "--epochs", 
                      type=int, default=2000,
                      help="number of epochs to train"
                      )

  parser.add_argument("-n", "--num_sets", 
                      type=int, default=20,
                      help="number of hyperparameter sets to generate"
                      )

  parser.add_argument("-s", "--save", 
                      action="store_true", 
                      help="save the trained model"
                      )

  parser.add_argument("-l", "--log", 
                      action="store_true", 
                      help="log various details of training process and model"
                      )
  return parser

def _generate_random_learning_rate(lower_bound=0.0001, upper_bound=0.1):
  """
  Return a random learning rate in range (0.0001, 0.1)
  """
  lower = math.log10(lower_bound)
  upper = math.log10(upper_bound)

  r = random.uniform(lower, upper)
  return 10 ** r

def _generate_random_int(min_val, max_val, step):
  """
  Return a random integer in range [min_value, max_value], 
  incremented by step.
  """
  if min_val > max_val:
      raise ValueError("min_value must be less than or equal to max_value")

  if step <= 0:
      raise ValueError("step must be a positive integer")

  num_values = (max_val - min_val) // step + 1
  random_index = random.randint(0, num_values - 1)
  random_value = min_val + random_index * step

  return random_value

def _log(filename, message, mode="a"):
  """
  Add a line to a log file.
  """
  os.makedirs(os.path.dirname(filename), exist_ok=True)

  with open(filename, mode) as f:
    f.write(message + "\n")


# might still be useful for data wrangling and scripts
def load_data(dataset: str): 
  BASE_PATH = "data/"
  with open(f"{BASE_PATH}{dataset}.json", 'r', encoding='utf-8') as file:
      data_dict = json.load(file)
  return data_dict

# might still be useful for data wrangling and scripts
def pretty_print(data: dict):
  print(json.dumps(data, indent=4))

