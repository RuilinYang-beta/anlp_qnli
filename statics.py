"""
Constants used in the project.
"""
import torch
from enum import Enum, auto

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# notation of numbers, as constant 
class Notation(Enum):
    ORIGINAL = "ori"
    SCIENTIFIC = "sci"
    ORIGINAL_CHAR = "ori_char"
    SCIENTIFIC_CHAR = "sch_char"


class Token(Enum):
  UNK = auto()  # unknown token
  SEP = auto()  # separator token to separate two statements
  PADDING = auto()  # padding token


notation2key = {
  Notation.ORIGINAL: ["statement1", "statement2"],
  Notation.SCIENTIFIC: ["statement1_sci_10E", "statement2_sci_10E"],
  Notation.ORIGINAL_CHAR: ["statement1_char", "statement2_char"],
  Notation.SCIENTIFIC_CHAR: ["statement1_sci_10E_char", "statement2_sci_10E_char"]
}


labels = ["Entailment", "neutral", "contradiction"]