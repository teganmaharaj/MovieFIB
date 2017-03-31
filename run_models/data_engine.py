import RAB_tools
from collections import OrderedDict
import numpy as np
import common, glob, re

import theano

def shuffle(data):
    idx = range(len(data))
    np.random.shuffle(idx)
    return [data[index] for index in idx]


def clean_word(word):
    return re.sub(r'([^\w\s]|_)+(?=\s|$)', '', word)




class LSMDC:

    def clean_vocabulary(self, vocab, train_data, reduced=True):
        word_count = OrderedDict()
        def count_word(caption):
            for w in caption:
                #w = clean_word(w)
                if len(w) == 0:
                    continue
                if w in word_count:
                    word_count[w] += 1
                else:
                    word_count[w] = 1

        ### Find words present more than 3 times in the training set
        for data in train_data:
            count_word(data[2])
            count_word(data[3])
            count_word(data[4])
        new_vocab = OrderedDict()
        idx = 1
        for k, v in word_count.items():
            if reduced:
                if v > 3:
                    if len(k) < 4:
                        print k

                    new_vocab[idx] = k
                    idx += 1
            else:
                new_vocab[idx] = k
                idx += 1

        ### Unknow words
        new_vocab[0] = 'unk'
        return new_vocab


    def out_vocabulary(self, vocab, train_data):
        word_count = OrderedDict()
        def count_word(caption):
            for w in caption:
                #w = clean_word(w)
                if len(w) == 0:
                    continue
                if w in word_count:
                    word_count[w] += 1
                else:
                    word_count[w] = 1

        ### Find words present more than 50 times in the training set
        for data in train_data:
            count_word(data[2])
            count_word(data[3])
            count_word(data[4])
        new_vocab = OrderedDict()
        idx = 1
        for k, v in word_count.items():
            if v > 50:
                new_vocab[idx] = k
                idx += 1

        return new_vocab

    def __init__(self, options):
        print 'load LSMDC'

        self.options = options

        self.data = RAB_tools.load_pkl(self.options['data_path'])

        self.features_2d = RAB_tools.load_pkl(self.options['features_path'] + "/LSMDC_googlenetfeatures.pkl")
        self.features_3d = RAB_tools.load_pkl(self.options['features_path'] + "/LSMDC_c3d.pkl")

        self.vocab = self.data['vocab']

        np.random.seed(1234)



        ### FIXME Would need to clean vocab anyway
        if self.options.reduce_vocabulary:
            self.vocab  = self.clean_vocabulary(self.vocab, self.data['train'],
                                                self.options.reduce_vocabulary)
        else:
            vocab = OrderedDict()
            for i, w in enumerate(self.vocab):
                vocab[i+1] = w
            vocab[0] = 'unk'
            self.vocab = vocab
        self.len_vocab = len(self.vocab.keys())
        ### Reverse dictionary
        self.vocab_reverse = OrderedDict()
        for (k, v) in self.vocab.items():
            self.vocab_reverse[v] = k
        #self.len_vocab = len(self.vocab)


        if self.options.use_out_vocab:
            self.out_vocab =  self.out_vocabulary(self.data['vocab'], self.data['train'])

            self.out_vocab_reverse = OrderedDict()
            for (k, v) in self.out_vocab.items():
                self.out_vocab_reverse[v] = k



        def get_sub_video(videos, num):
            vid_id = OrderedDict()

            for v in videos:
                if v[1] in vid_id:
                    vid_id[v[1]].append(v)
                else:
                    vid_id[v[1]] = [v]

            res = []
            for i, (k, v) in enumerate(vid_id.items()):
                if i >= num:
                    break
                else:
                    res.extend(v)
            return res

        self.train_list = self.data['train']
        if self.options.max_train_example is not None:
            self.train_list = get_sub_video(self.train_list, self.options.max_train_example)

        self.train = shuffle(self.train_list)
        self.valid = shuffle(self.data['val'])
        self.test = shuffle(self.data['test'])

        #for data in self.test:
        RAB_tools.dump_pkl(self.test, 'test.pkl')



        self.kf_train = RAB_tools.generate_minibatch_idx(len(self.train), self.options.batch_size)
        self.kf_valid = RAB_tools.generate_minibatch_idx(len(self.valid), self.options.batch_size)
        self.kf_test = RAB_tools.generate_minibatch_idx(len(self.test), self.options.batch_size)


    def send_request(self, whichset):
        assert whichset in ['update', 'train', 'valid', 'test']
        if whichset == 'update':
            self.train = shuffle(self.train_list)
            self.kf_train = RAB_tools.generate_minibatch_idx(len(self.train), self.options.batch_size)
            self.mb_update = 0
        else:
            self.mb = 0



    def fetch_vid_feats(self, vid_ids, augmented=False, repeat_padding=False):
        ### FIXME add repeat padding

        if self.options.features_type == "2D":
            features = self.features_2d
        else:
            features = self.features_3d

        def get_frames(nb_frames, augmented=False):
            if augmented:
                ### Subsampling
                sub = np.floor(max(1, (nb_frames) / float(self.options.n_subframes)))
                if sub > 1:
                    sub = np.random.randint(1, sub)
                ### Start/endFrames
                start = 0
                if nb_frames - sub * self.options.n_subframes + 1 > 0:
                    start =  np.random.randint(0,
                                               nb_frames - sub * self.options.n_subframes + 1)
                end = min(nb_frames, start + sub*self.options.n_subframes)
            else:
                sub = max(1, np.floor((nb_frames) / float(self.options.n_subframes)))
                start = 0
                end = min(nb_frames, sub*self.options.n_subframes)
            return start, end, sub

        size = (len(vid_ids), self.options.n_subframes, self.options.input_dim)
        feats = np.zeros(size).astype(theano.config.floatX)
        masks = np.zeros(size[:2]).astype(theano.config.floatX)
        for i, vid_id in enumerate(vid_ids):
            try:
                cur_feat = features[vid_id]
            except:
                print "Missing", vid_id
                continue
            masks[i, :cur_feat.shape[0]] = 1.
            start, end, sub = get_frames(cur_feat.shape[0], augmented=augmented)
            feats[i, :min(feats.shape[1], cur_feat.shape[0]), :] = cur_feat[start:end:sub, :]
        return feats, masks

    def fetch_vid_feats_fuse(self, vid_ids, augmented=False, repeat_padding=False):
        ### FIXME add repeat padding

        def get_frames_3d(nb_frames):
            start = 0
            end = nb_frames
            if end > self.options.n_subframes:
                end = self.options.n_subframes
            return start, end, 1

        def get_frames_2d(nb_frames, nb_seg, augmented=False):
            if augmented:
                ### Subsampling
                sub = 5 ## 16 /3 as there is a subsampling of 3 for 2d features
                start = 0
                if nb_frames - sub * nb_seg + 1 > 0:
                    start =  np.random.randint(0, nb_frames - sub * nb_seg + 1)
                end = min(nb_frames, start + sub*nb_seg)
            else:
                sub = 5
                start = 0
                end = min(nb_frames, sub*nb_seg)
            return start, end, sub

        size = (len(vid_ids), self.options.n_subframes, self.options.input_dim)
        feats = np.zeros(size).astype(theano.config.floatX)
        masks = np.zeros(size[:2]).astype(theano.config.floatX)

        # for i, vid_id in enumerate(self.features_3d.keys()):
        #     print i, vid_id, self.features_3d[vid_id].shape[0], self.features_2d[vid_id].shape[0]
        #     if 16*self.features_3d[vid_id].shape[0] > self.features_2d[vid_id].shape[0]:
        #         import pdb; pdb.set_trace()

        for i, vid_id in enumerate(vid_ids):
            try:
                cur_feat = self.features_3d[vid_id]
            except:
                print "Missing", vid_id
                continue

            start3d, end3d, sub = get_frames_3d(cur_feat.shape[0])
            cur_feat_3d = cur_feat[start3d:end3d, :]

            cur_feat_2d = self.features_2d[vid_id]
            start2d, end2d, sub2d = get_frames_2d(cur_feat_2d.shape[0], cur_feat_3d.shape[0], augmented=augmented)
            cur_feat_2d = cur_feat_2d[start2d:end2d:sub2d, :]


            if cur_feat_3d.shape[0] != cur_feat_2d.shape[0]:
                cur_feat_3d = cur_feat_3d[:cur_feat_2d.shape[0], :]

            #print cur_feat_3d.shape,  cur_feat_2d.shape, start2d, end2d, sub2d
            feats[i, :min(feats.shape[1], cur_feat_3d.shape[0]), :] = np.concatenate([cur_feat_3d, cur_feat_2d], axis=1)
            masks[i, :cur_feat_3d.shape[0]] = 1.

        return feats, masks


    def fetch_text(self, example):
        #ID, vidID, fillin, target = example
        #ID_vidID_fillin_target[ID] = [vidID, fillin, target]


        def get_index(word, vocab=self.vocab_reverse):
            try:
                return vocab[word]
            except:
                return 0

        # def get_index(word, vocab=None):
        #    if word in self.vocab:
        #        return self.vocab.index(word)
        #    else:
        #        return 0

        PRE = []
        POST = []
        Y = []
        for i, value in enumerate(example):
            num, vid_id, pre, post, blank = value

            pre_idxs = []
            for word in pre:
                pre_idxs.append(get_index(word))
            post_idxs = []
            for word in post:
                post_idxs.append(get_index(word))

            PRE.append(pre_idxs)
            POST.append(post_idxs)

            if self.options.use_out_vocab:
                Y.append(get_index(blank, vocab=self.out_vocab_reverse))
            else:
                Y.append(get_index(blank))

        maxlen = np.max([len(c) for c in PRE+POST]) + 1

        #PRE onehots and mask
        pre = np.zeros((len(PRE), maxlen)).astype('int32')
        pre_mask = np.zeros((len(PRE), maxlen)).astype(theano.config.floatX)
        for i, example in enumerate(PRE):
            pre[i, :len(example)] = example
            pre_mask[i][:len(example)] = 1

        #POST onehots and mask
        post = np.zeros((len(POST), maxlen)).astype('int32')
        post_mask = np.zeros((len(POST), maxlen)).astype(theano.config.floatX)
        for i, example in enumerate(POST):
            post[i, :len(example)] = example
            post_mask[i][:len(example)] = 1

        #Target
        y = np.array(Y).astype('int32')

        return pre, pre_mask, post, post_mask, y


    def fetch_mb(self, whichset):
        data = self.train
        kf = self.kf_train

        if whichset == 'update':
            mb = self.mb_update
            self.mb_update += 1
            augmentation = True
        else:
            mb = self.mb
            self.mb += 1
            augmentation = False
        if whichset == 'valid':
            data = self.valid
            kf = self.kf_valid
        elif whichset == 'test':
            data = self.test
            kf = self.kf_test

        if mb >= len(kf):
            return [], True

        mb_data = [data[i] for i in kf[mb]]
        vid_ids = [c[1] for c in mb_data]

        ### Get video features
        if self.options.features_type  == "Fuse":
            features, features_mask = self.fetch_vid_feats_fuse(vid_ids, augmented=augmentation)
        else:
            features, features_mask = self.fetch_vid_feats(vid_ids, augmented=augmentation)

        ### Get text features
        (before, before_mask,
         after, after_mask,
         target) = self.fetch_text(mb_data)


        minibatch = [features, features_mask,
                     before, before_mask,
                     after, after_mask,
                     target]

        return minibatch, False

    def get_test_value(self):
        mb = self.options.batch_size
        k = self.options.n_subframes

        x = np.random.uniform(size=(mb, k, self.options.input_dim)).astype('float32')
        x_mask = np.ones((mb, k)).astype('float32')

        pre = np.ones((mb, k)).astype('int32')
        pre_mask = np.ones((mb, k)).astype('float32')

        post = np.ones((mb, k)).astype('int32')
        post_mask = np.ones((mb, k)).astype('float32')

        y = np.random.randint(0, 100, size=(mb)).astype('int32')

        return x, x_mask, pre, pre_mask, post, post_mask, y



if __name__ == '__main__':
    from config import options
    d = LSMDC(options)
    d.send_request('test')
    epoch = False
    while not epoch:
        mb, epoch  = d.fetch_mb('test')
