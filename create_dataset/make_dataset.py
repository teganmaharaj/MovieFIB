import nltk, re

stop_list = ['SOMEONE',
             'is',
             'there', 
             'new', 
             'up', 
             'now',
             'Now',
             'later',
             'Later',
             'then'
             'before',
             'Mr', 
             'Mrs',
             'SOMEONE\'s',
             'it',
             'not',
             'Meanwhile',
             'meanwhile',
             'while',
             'it\'s',
             'them',
             'something',
             'aside',
             'be',
             'SOMEONE\'',
             'as',
             'who',
             'who\'s',
             'in',
             'THE',
             'it\'s',
             'does',
             'doesn\'t',
             'other',
             'other\'s',
             'can\'t',
             'on',
             'Elsewhere',
             'OF',
             'A',
             'a',
             'Dr',
             'IN',
             'off',
             'out',
             '"',
             ]
with open ("clipID_anno_FI_B.csv", 'w') as datafile:
    with open ("MVADalignedTrainVal_annos_upd2.csv", 'rb') as clip_annos:
        for line in clip_annos:
            cols = line.split('\t')
            anno = cols[-1].strip().strip('\.')
            clip_info = '\t'.join(cols[:-1])
            words = anno.split()
            pos_tags = nltk.pos_tag(words)
            contains_blankable = False 
            for i,word_tag_pair in enumerate(pos_tags):
                POS = word_tag_pair[1][0]
                blank = word_tag_pair[0].strip(',').strip(':').strip('\"')
                tag = word_tag_pair[1]
                if POS=='V' or POS=='J' or POS=='N' or POS=='R':
                    if blank not in stop_list and not blank=='':
                        #print blank + tag
                        contains_blankable = True
                        fill_in_parts = anno.split(blank)
                        fill_in = fill_in_parts[0]+' _____ '+fill_in_parts[1]
                        things_to_write = [clip_info,
                                        anno,
                                        str(pos_tags),
                                        fill_in,
                                        blank,
                                        tag]
                        datafile.write('\t'.join(things_to_write)+'\n')
            if contains_blankable == False:
                things_to_write = [clip_info,
                                   anno,
                                   str(pos_tags),
                                   fill_in,
                                   'NOTHINGTOBLANK',
                                   'NOTHINGTOBLANK']
                datafile.write('\t'.join(things_to_write)+'\n')
