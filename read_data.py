from pprint import pprint
def read_data(file):
    sentences_tags=[]
    with open (file,'r') as f:
        sentence=[]
        tags=[]
        for idx,line in enumerate(f):
            if idx<2:continue
            if len(line.strip('\n'))==0:
                sentences_tags.append((sentence,tags))
                sentence = []
                tags = []
            else:
                row=line.split(" ")
                word,pos1,pos2,ner=row
                sentence.append(word)
                tags.append(ner.strip('\n'))

    return sentences_tags

