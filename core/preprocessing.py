import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except:
    pass

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))
_stop_words.update({
    'patient', 'also', 'well', 'mr', 'ms', 'dr', 'upon', 'per',
    'day', 'days', 'mg', 'ml', 'noted', 'given', 'would', 'one',
    'two', 'po', 'sig', 'bid', 'tid', 'prn', 'tab', 'tablet',
    'disp', 'refill', 'name', 'date', 'hct', 'hgb', 'rbc', 'wbc',
    'mch', 'mchc', 'mcv', 'rdw', 'na', 'cl', 'hco', 'bun', 'cr'
})

# NLTK's default English stopword list includes negation terms (no, not, nor,
# ...). Clinical NLP literature on MIMIC-IV specifically flags this: negation
# and uncertainty carry the actual clinical meaning in a note ("no chest
# pain" vs "chest pain" are opposite findings), and roughly 13% of clinical
# concepts in notes appear in a negated context, so blanket stopword removal
# silently flips or erases meaning. Keep these out of the stopword set.
_NEGATION_TERMS = {
    'no', 'not', 'nor', 'never', 'none', 'neither', 'without', 'cannot',
    "don't", "didn't", "doesn't", "isn't", "wasn't", "weren't", "won't",
    "wouldn't", "shouldn't", "couldn't", "can't", "aren't", 'denies', 'denied'
}
_stop_words -= _NEGATION_TERMS

def clean_for_viz(text):
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens
              if t.isalpha() and t not in _stop_words
              and (len(t) > 2 or t in _NEGATION_TERMS)]
    return ' '.join(tokens)

from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_note(text):
    if not isinstance(text, str):
        return ''
    # Step 1: Remove de-identification placeholders
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    # Step 2: Remove lines that are only underscores or whitespace
    text = re.sub(r'^[_\s]+$', '', text, flags=re.MULTILINE)
    # Step 3: Lowercase
    text = text.lower()
    # Step 4: Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Step 5: Keep only letters, numbers, and basic punctuation
    text = re.sub(r'[^a-z0-9\s.,;:?!]', ' ', text)
    # Step 6: Tokenise
    tokens = word_tokenize(text)
    # Step 7: Remove stop words and short tokens (but keep short negation
    # terms like "no" -- see _NEGATION_TERMS above)
    tokens = [t for t in tokens if t.isalpha() and t not in _stop_words
              and (len(t) > 2 or t in _NEGATION_TERMS)]
    # Step 8: Lemmatise with POS tagging
    pos_tags = nltk.pos_tag(tokens)
    tokens = [_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(tokens)

if __name__ == '__main__':
    print("Preprocessing functions loaded successfully.")
