import spacy

text = 'Noam Chomsky is a linguist. BOOIKNOIHHOOL'
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.is_digit, token.is_oov, token.is_stop)