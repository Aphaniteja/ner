def vocab_to_idx(sentence_tags):
    vocabtoidx = {}
    labelstoidx = {}
    vocab_idx = 1
    label_idx = 1
    for sentence_tag in sentence_tags:
        for word, tag in zip(*(sentence_tag)):
            if word not in vocabtoidx:
                vocabtoidx[word] = vocab_idx
                vocab_idx += 1
            if tag not in labelstoidx:
                labelstoidx[tag] = label_idx
                label_idx += 1
    vocabtoidx['UNK'] = vocab_idx + 1
    vocabtoidx['pad']=0
    labelstoidx['pad']=0
    return vocabtoidx, labelstoidx


def prepare_senetence_tags(sentence_tags, vocabtoidx, labelstoidx):
    sentence = []
    tags = []
    for word, tag in zip(*sentence_tags):
        if word in vocabtoidx:
            sentence.append(vocabtoidx[word])
        else:
            sentence.append(vocabtoidx['UNK'])
        tags.append(labelstoidx[tag])

    return sentence, tags


def prepare_batch(sentences_tags, vocabtoidx, labelstoidx):
    for sentence_tag in sentences_tags:
        yield prepare_senetence_tags(sentence_tag, vocabtoidx, labelstoidx)


def word_embeddings():
    pass