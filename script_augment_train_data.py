import json
from collections import Counter

from statics import labels
from utils import load_data, pretty_print

pairs_to_swap = [
  ('statement1', 'statement2'), 
  ('statement1_sci_10E', 'statement2_sci_10E'), 
  ('statement1_char', 'statement2_char'), 
  ('statement1_sci_10E_char', 'statement2_sci_10E_char')
]

def get_augmented_record(record):
  e, n, c = labels

  if record['answer'] == e:
    return [record]

  elif record['answer'] == n or record['answer'] == c:
    return [record, get_swapped_record(record)]

  else: 
    raise ValueError("Invalid label")


def get_swapped_record(record):
  record_deepcopy = json.loads(json.dumps(record))

  for key1, key2 in pairs_to_swap:
    record_deepcopy[key1], record_deepcopy[key2] = record_deepcopy[key2], record_deepcopy[key1]

  return record_deepcopy



train = load_data('train')

freq_count_before = Counter([record['answer'] for record in train])
print("=== before ===")
print(freq_count_before)

train_augmented = [aug_record for record in train for aug_record in get_augmented_record(record) ]

freq_count_after = Counter([record['answer'] for record in train_augmented])
print("=== after ===")
print(freq_count_after)

file_path = './data/train_augmented.json'
with open(file_path, 'w') as file:
    json.dump(train_augmented, file, indent=2)

print('======')
print(f'Data has been written to {file_path}')
