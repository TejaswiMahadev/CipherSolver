# CipherSolver & Cryptography Tool

This is a cryptography tool built using Streamlit, designed to help users encrypt and decrypt text using popular cipher methods such as Caesar Cipher, Vigenere Cipher, Rail Fence Cipher, and Simple Columnar Transposition. Additionally, it includes functionality for frequency analysis of n-grams in the given text, word cloud generation, and a basic anagram solver.

## Features
**Encryption & Decryption:**
- Caesar Cipher
-  Vigenere Cipher
-  Rail Fence Cipher
-  Simple Columnar Transposition
**Frequency Analysis:**
- Letter Frequency Analysis
- Digraph and Trigraph Frequency Analysis
- Visualization with bar charts, pie charts, and word clouds
**Anagram Solver:**
- Generate Anagrams from a given word or phrase
- Check if two words or phrases are anagrams of each other
**Synonyms & Antonyms:**
- Get synonyms and antonyms for any word using WordNet from NLTK

## Requirements
- Python 3.x
- Streamlit
- NLTK
- Spacy
- Plotly
- Pandas
- WordCloud
- Matplotlib

## Functions
**Encryption & Decryption Methods**
- Caesar Cipher: A substitution cipher where each letter is shifted by a fixed number of positions.
- Vigenere Cipher: A polyalphabetic cipher that uses a keyword to shift letters.
- Rail Fence Cipher: A transposition cipher that arranges text in a zigzag pattern.
- Simple Columnar Transposition: A method where text is written in columns and then rearranged.
  
**Frequency Analysis**
- Analyze the frequency of individual letters, digraphs (pairs of letters), and trigraphs (triplets of letters) in a given text.
- Visualize the results using bar charts, pie charts, and word clouds.
  
**Anagram Solver**
- Generate Anagrams: Given a word or phrase, generate all possible anagrams.
- Check Anagram Pair: Check if two given words or phrases are anagrams of each other.
  
**Synonyms & Antonyms**
- WordNet Synonyms & Antonyms: Given a word, retrieve a list of its synonyms and antonyms using the WordNet corpus from NLTK.

## Contributing
Feel free to fork the repository, make improvements, and create pull requests. Any suggestions and improvements are welcome.

## License
This project is open-source and available under the MIT License.
