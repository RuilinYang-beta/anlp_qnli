"""
Constants used in the project.
"""

from enum import Enum, auto

# notation of numbers, as constant 
class Notation(Enum):
    ORIGINAL = auto()
    SCIENTIFIC = auto()
    ORIGINAL_CHAR = auto()
    SCIENTIFIC_CHAR = auto()


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