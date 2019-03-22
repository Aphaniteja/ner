def read_data(file):
    sentences_tags = []
    with open(file, 'r') as f:
        sentence = []
        tags = []
        for idx, line in enumerate(f):
            if len(line.strip('\n')) == 0:
                sentences_tags.append((sentence, tags))
                sentence = []
                tags = []
            else:
                row = line.split(" ")
                word, pos1, pos2, ner = row
                sentence.append(word)
                ner = ner.strip('\n')
                tags.append(ner)

    return sentences_tags

