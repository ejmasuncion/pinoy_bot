"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List
from sklearn import tree
import numpy as np

FILIPINO_BIGRAMS = ['ng', 'ka', 'in', 'ay', 'um', 'mag', 'nag', 'may', 'na', 'sa', 'ol','nya']
ENGLISH_BIGRAMS = ['th', 'sh', 'ch', 'wh', 'ed', 'ly', 'er', 'es', 'ou', 'ea', 'io', 'al', 'is', 'at', 'an', 'he', 'lk', 'tr']

def count_filipino_bigrams(word: str, target_bigrams: list) -> int:
    word_lower = str(word).lower()
    total_count = 0
    
    for bigram in target_bigrams:
        total_count += word_lower.count(bigram)
            
    return total_count

def vowel_word_ratio_feature(word):
    if not isinstance(word, str):
        return 0.0
    
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    num_vowels = sum(1 for ch in word if ch.isalpha() and ch in vowels)
    num_consonants = sum(1 for ch in word)

    if num_consonants == 0:
        return 1.0 if num_vowels > 0 else 0.0
    return num_vowels / num_consonants

def count_english_bigrams(word: str, target_bigrams: list) -> int:
    word_lower = str(word).lower()
    total_count = 0
    
    for bigram in target_bigrams:
        # Count all non-overlapping occurrences of the bigram in the word
        total_count += word_lower.count(bigram)
            
    return total_count

def check_foreign_alphabet(word: str) -> int:
    foreign_letters = ['c', 'f', 'j', 'q', 'v', 'x', 'z']
    word_lower = str(word).lower()

    for letter in word_lower:
        if letter in foreign_letters:
            return 1
    return 0

def tagalog_xfix_check(word : str) -> int:
    prefix = tuple(['ka' ,'pang', 'taga', 'mag', 'nag'])
    affix = tuple(['an', 'ng', 'ero', 'era'])

    if (word.startswith(prefix) or word.endswith(affix)):
        return 1
    else: return 0

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)
    with open('my_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
    features = []
    for token in tokens:
        # Ensure the token is treated as a string, especially for punctuation/numbers
        word = str(token)
        
        # Calculate all features for the current token
        filipino_bigram_count = count_filipino_bigrams(word, FILIPINO_BIGRAMS)
        english_bigram_count = count_english_bigrams(word, ENGLISH_BIGRAMS)
        vowel_word_ratio = vowel_word_ratio_feature(word)
        has_foreign_alphabet = check_foreign_alphabet(word)
        has_tagalog_xfix = tagalog_xfix_check(word)
        
        # Combine into a feature vector (must match the order used during training!)
        feature_vector = [
            has_tagalog_xfix,
            vowel_word_ratio,
            filipino_bigram_count,
            english_bigram_count,
            has_foreign_alphabet,
        ]
        features.append(feature_vector)

    features = np.array(features)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)
    predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]
    tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    return tags

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print(tags)