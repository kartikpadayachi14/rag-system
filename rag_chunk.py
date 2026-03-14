from nltk.tokenize import sent_tokenize

def smart_chunk(text, max_sentences =2):
    sentences = sent_tokenize(text)
    chunks= []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks

text = """Transformers are neural networks.
They were introduced in 2017.
They revolutionized NLP.
They allow parallel processing."""

print(smart_chunk(text))