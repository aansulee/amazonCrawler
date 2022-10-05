import nltk



class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return conlltags

"""
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
chunker = BigramChunker(train_sents)
chunks = chunker.parse(nltk.pos_tag(nltk.word_tokenize("Mac Mini (stylized as Mac mini) is a small form-factor desktop computer developed and marketed by Apple Inc.")))
print(chunks)

for word, pos, chunktag in chunks:
    if chunktag == "B-NP":
        np = word
    if chunktag == "I-NP":
        np += " " + word
    if chunktag == "O":
        print(np)

for syn in wn.synsets("computer")[0].part_meronyms():
    print(syn.lemma_names())
"""