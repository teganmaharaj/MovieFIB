import sys

filename = sys.argv[1]
blanks = []
vocab = {}
POSs = []
POSvocab = {}
blank_POS_vocab = []
blank_POS_vocab = {}
with open (filename, 'rb') as f:
    for line in f:
        cols = [x for x in line.strip().split('\t')]
        blank = str(cols[-2].strip(',').strip('"'))
        POS = str(cols[-1])
        blank_POS = blank+'\t'+POS
        blanks.append(blank)
        POSs.append(POS)
        if blank in vocab:
            vocab[blank]+=1
        else: 
            vocab[blank]=1
        if POS in POSvocab:
            POSvocab[POS]+=1
        else:
            POSvocab[POS]=1
        if blank_POS in blank_POS_vocab:
            blank_POS_vocab[blank_POS]+=1
        else:
            blank_POS_vocab[blank_POS]=1
with open ('vocab_'+filename.split('.')[0]+'.txt', 'w') as f2:
    vocablist = sorted(vocab, key=vocab.get, reverse=True)
    for elem in vocablist:
        f2.write(elem +'\t'+ str(vocab[elem])+'\n')
with open ('POSvocab_'+filename.split('.')[0]+'.txt', 'w') as f3:
    POSlist = sorted(POSvocab, key=POSvocab.get, reverse=True)
    for elem in POSlist:
        f3.write(elem +'\t'+ str(POSvocab[elem])+'\n')
with open ('blank_POSvocab_'+filename.split('.')[0]+'.txt', 'w') as f4:
    blankPOSlist = sorted(blank_POS_vocab, key=blank_POS_vocab.get, reverse=True)
    for elem in blankPOSlist:
        f4.write(elem +'\t'+ str(blank_POS_vocab[elem])+'\n')
        