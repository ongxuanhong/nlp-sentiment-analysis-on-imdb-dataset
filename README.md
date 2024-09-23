# NLP sentiment analysis on IMDB dataset

Text preprocessing is an essential step in Natural Language Processing (NLP) to prepare raw text data for further analysis, modeling, or machine learning. Here are the key text preprocessing techniques:

### 1. **Lowercasing**
   - Convert all text to lowercase to ensure uniformity and avoid treating words like "Apple" and "apple" as different.
   - Example: "Hello World" → "hello world"

### 2. **Tokenization**
   - Split the text into smaller units like words, sentences, or subwords.
   - Example: "I love NLP" → ["I", "love", "NLP"]

### 3. **Stopwords Removal**
   - Remove common words (e.g., "and", "the", "is") that don’t contribute much to the meaning of a sentence.
   - Example: "This is a simple example" → ["simple", "example"]

### 4. **Punctuation Removal**
   - Remove punctuation marks as they usually do not contribute much meaning (though this depends on the task).
   - Example: "Hello, World!" → "Hello World"

### 5. **Lemmatization**
   - Convert words to their base or dictionary form, taking grammatical rules into account.
   - Example: "better" → "good", "running" → "run"

### 6. **Stemming**
   - Strip words to their base form by removing suffixes and prefixes.
   - Example: "caring", "cars" → "car"

### 7. **Removing Numbers**
   - Remove numerical digits from the text, unless the numbers have significance for the analysis.
   - Example: "In 2024, AI will dominate" → "AI will dominate"

### 8. **Removing Special Characters**
   - Remove symbols like $, %, @, etc., unless they are relevant.
   - Example: "I paid $100!" → "I paid 100"

### 9. **Removing URLs and HTML Tags**
   - Strip out any web links or HTML tags from the text.
   - Example: "Visit <a href='https://example.com'>this site</a>" → "Visit this site"

### 10. **Text Normalization**
   - Normalize text by expanding contractions (e.g., "don't" → "do not") or handling spelling corrections.
   - Example: "can't" → "cannot", "thx" → "thanks"

### 11. **Handling Misspellings**
   - Correct common spelling errors in the text.
   - Example: "recieve" → "receive"

### 12. **Part-of-Speech (POS) Tagging**
   - Tag each word with its corresponding part of speech (noun, verb, etc.), which can be useful for more advanced NLP tasks.
   - Example: "Running is fun" → [("Running", "VBG"), ("is", "VBZ"), ("fun", "NN")]

### 13. **Text Vectorization**
   - Convert text into numerical representation for machine learning algorithms. Common techniques include:
     - **Bag of Words (BoW)**: Represent the document by counting the occurrence of each word.
     - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weighs terms by their frequency and uniqueness across documents.
     - **Word Embeddings**: Represent words as dense vectors (e.g., Word2Vec, GloVe).

### 14. **Sentence Segmentation**
   - Divide text into meaningful sentences, which is especially useful for tasks like summarization or parsing.

### 15. **Handling Repeated Characters**
   - Reduce repeated characters in text.
   - Example: "sooooo good" → "so good"

### 16. **Handling Emoji and Emoticons**
   - Depending on the task, emojis can be removed, replaced with text (e.g., 😊 → "happy"), or analyzed as features.

### 17. **Handling Slang and Abbreviations**
   - Expand slang or abbreviations for uniformity.
   - Example: "OMG" → "Oh my God", "BTW" → "By the way"
