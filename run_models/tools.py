def visualize(save_model_dir, counter, record, video, state, memory,
              delta_c, delta_c_2, g_i, g_f, g_o, gg, xW_b, xW_hU_b,
              grad_dummy_h, grad_dummy_c, grad_dummy_xW_hU_b,
              W_i, W_o, W_f, W_g, U_i, U_o, U_f, U_g):
    print 'conv lstm visualize...'
    import matplotlib.pyplot as plt
    idx = 0
    def d2d(x):
        a,b,c,d = x.shape
        return x.reshape(a,b*c*d)
    video = d2d(video[idx]) # (m,t,c,x,y) -> (t,c,x,y) -> (t,c*x*y)
    hU = d2d((xW_hU_b - xW_b)[:,idx])
    xW_b = d2d(xW_b[:,idx])
    xW_hU_b = d2d(xW_hU_b[:,idx])
    state = d2d(state[:,idx])
    memory = d2d(memory[:,idx])
    delta_c = d2d(delta_c[:,idx])
    delta_c_2 = d2d(delta_c_2[:,idx])
    g_i = d2d(g_i[:,idx])
    g_f = d2d(g_f[:,idx])
    g_o = d2d(g_o[:,idx])
    gg = d2d(gg[:,idx])
    grad_dummy_h = d2d(grad_dummy_h[:,idx])
    grad_dummy_c = d2d(grad_dummy_c[:,idx]) # (t,c*x*y)
    grad_dummy_xW_hU_b = grad_dummy_xW_hU_b[:,idx] # (t,4*c,x,y)
    W_i = d2d(W_i.get_value()) # (c,c*x*y)
    W_o = d2d(W_o.get_value())
    W_f = d2d(W_f.get_value())
    W_g = d2d(W_g.get_value())
    U_i = d2d(U_i.get_value())
    U_o = d2d(U_o.get_value())
    U_f = d2d(U_f.get_value())
    U_g = d2d(U_g.get_value())

    fig= plt.figure(figsize=(30,15))
    ax = plt.subplot2grid((5, 8), (0, 0))
    img = ax.imshow(state, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('h')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (0, 1))
    img = ax.imshow(memory, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('c')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (0, 2))
    img = ax.imshow(delta_c, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('delta_c')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (0, 3))
    img = ax.imshow(delta_c_2, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('delta_c_2')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (0, 4))
    img = ax.imshow(g_i, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('input gate')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (0, 5))
    img = ax.imshow(g_f, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('forget gate')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (0, 6))
    img = ax.imshow(g_o, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('output gate')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (0, 7))
    img = ax.imshow(gg, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('input part of tanh(xW+hU+b)')
    ax.set_ylabel('time')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (1, 0))
    norm_g_i, norm_g_f, norm_g_o, norm_gg = numpy.split(
        grad_dummy_xW_hU_b, 4, axis=1)
    ax.plot(numpy.sqrt((norm_g_i**2).sum(1).sum(1).sum(1)), 'r', label='g_i')
    ax.plot(numpy.sqrt((norm_g_f**2).sum(1).sum(1).sum(1)), 'g', label='g_f')
    ax.plot(numpy.sqrt((norm_g_o**2).sum(1).sum(1).sum(1)), 'b', label='g_o')
    ax.plot(numpy.sqrt((norm_gg**2).sum(1).sum(1).sum(1)), 'k', label='g')
    ax.legend()
    ax.set_title('grad w.r.t hU+xW+b')
    ax.set_ylabel('L2 norm')
    ax.set_xlabel('time')

    ax = plt.subplot2grid((5, 8), (1, 1))
    norm = numpy.sqrt((grad_dummy_c**2).sum(1))
    img = ax.plot(norm, '.-')
    ax.set_title('grad c')
    ax.set_ylabel('L2 norm')
    ax.set_xlabel('time')

    ax = plt.subplot2grid((5, 8), (1, 2))
    ax.plot(numpy.sqrt((grad_dummy_h**2).sum(1)))
    ax.set_title('grad w.r.t h')
    ax.set_ylabel('L2 norm')
    ax.set_xlabel('time')
    
    ax = plt.subplot2grid((5, 8), (1, 3))
    ax.plot((numpy.sqrt(delta_c_2**2).sum(1)), '.-')
    ax.set_title('delta_c_2')
    ax.set_ylabel('L2')
    ax.set_xlabel('time')

    ax = plt.subplot2grid((5, 8), (1, 4))
    ax.plot((numpy.sqrt(state**2).sum(1)), '.-')
    ax.set_title('h')
    ax.set_ylabel('L2')
    ax.set_xlabel('time')
    
    ax = plt.subplot2grid((5, 8), (1, 5))
    ax.plot((numpy.sqrt(memory**2).sum(1)), '.-')
    ax.set_title('c')
    ax.set_ylabel('L2')
    ax.set_xlabel('time')

    ax = plt.subplot2grid((5, 8), (1, 6))
    ax.plot((numpy.sqrt(delta_c**2).sum(1)), '.-')
    ax.set_title('delta_c')
    ax.set_ylabel('L2')
    ax.set_xlabel('time')

    ax = plt.subplot2grid((5, 8), (1, 7))        
    img = ax.imshow(video, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('input video')
    ax.set_ylabel('time')
    plt.colorbar(img)

    def sample_at_k_and_errorbar(x, positions, prefix, k=250,
                                 names=['g_i', 'g_f', 'g_o', 'g']):
        assert len(positions) == 4
        t1, t2, t3, t4 = numpy.split(x, 4, axis=1)
        if t1.shape[1] > k:
            use_idx = [idx[0] for idx in numpy.array_split(numpy.arange(t1.shape[1]), k)]
            activations = [t1[:,use_idx], t2[:,use_idx], t3[:,use_idx], t4[:,use_idx]]
        else:
            activations = [t1, t2, t3, t4]
        for activation, position, name in zip(activations, positions, names):
            std = activation.std(0)
            mean = activation.mean(0)
            ax = plt.subplot2grid((5, 8), position)
            ax.errorbar(range(std.shape[0]), mean, std, fmt='o')
            ax.set_title('%s, %s part'%(prefix, name))
            ax.set_xlabel('dim, sampled at %d'%k)
            ax.grid(True)
    
    sample_at_k_and_errorbar(xW_hU_b, [(2,0), (2,1), (2,2), (2,3)], 'xW+hU+b')
    sample_at_k_and_errorbar(hU, [(2,4), (2,5), (2,6), (2,7)], 'hU')
    sample_at_k_and_errorbar(xW_b, [(3,0), (3,1), (3,2), (3,3)], 'xW+b')

    idx = RAB_tools.shuffle_idx(video.shape[1])
    idx = idx[:int(video.shape[1] * 0.25)]
    video_ = video[:, idx]
    ax = plt.subplot2grid((5, 8), (3, 4), colspan=3)
    std =video_.std(0)
    mean = video_.mean(0)
    ax.errorbar(range(std.shape[0]), mean, std, fmt='o')
    ax.set_title('video inputs')
    ax.set_xlabel('dim, random 0.25')
    ax.grid(True)

    ax = plt.subplot2grid((5, 8), (3, 7))
    img = ax.plot(record, '.-')
    ax.set_title('cost (moving avg)')
    ax.set_xlabel('number of updates')

    ax = plt.subplot2grid((5, 8), (4, 0))
    img = ax.imshow(W_i, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('W_lstm_g_i')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (4, 1))
    img = ax.imshow(W_f, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('W_lstm_g_f')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (4, 2))
    img = ax.imshow(W_o, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('W_lstm_g_o')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (4, 3))
    img = ax.imshow(W_g, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('W_lstm_g_g')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (4, 4))
    img = ax.imshow(U_i, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('U_lstm_g_i')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (4, 5))
    img = ax.imshow(U_o, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('U_lstm_g_o')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (4, 6))
    img = ax.imshow(U_f, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('U_lstm_g_f')
    plt.colorbar(img)

    ax = plt.subplot2grid((5, 8), (4, 7))
    img = ax.imshow(U_g, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('U_lstm_g_g')
    plt.colorbar(img)

    plt.tight_layout()
    plt.savefig(save_model_dir+'conv_lstm_mb_%d.png'%counter)
    plt.close()
