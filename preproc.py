import torch


def vocab_to_idx(sentence_tags):
    vocabtoidx = {}
    labelstoidx = {}
    vocab_idx = 1
    label_idx = 1
    for sentence_tag in sentence_tags:
        for word, tag in zip(*sentence_tag):
            if word not in vocabtoidx:
                vocabtoidx[word] = vocab_idx
                vocab_idx += 1
            if tag not in labelstoidx:
                labelstoidx[tag] = label_idx
                label_idx += 1
    vocabtoidx['UNK'] = vocab_idx + 1
    vocabtoidx['pad'] = 0
    labelstoidx['pad'] = 0
    return vocabtoidx, labelstoidx


def prepare_sentence_tags(sentence_tags, vocabtoidx, labelstoidx):
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
        yield prepare_sentence_tags(sentence_tag, vocabtoidx, labelstoidx)


def word_embeddings(vocabtoidx):
    import io

    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            if tokens[0] in vocabtoidx or tokens[0] == 'UNK':
                data[tokens[0]] = list(map(float, tokens[1:]))
        return data

    word_vecs = load_vectors('data/wiki-news-300d-1M.vec')
    weights_matrix = torch.zeros((len(vocabtoidx) + 1, 300))

    for word, idx in vocabtoidx.items():
        try:
            weights_matrix[idx] = torch.tensor((word_vecs[word]))
        except KeyError:
            weights_matrix[idx] = torch.tensor((word_vecs['UNK']))

    return weights_matrix
