def set_template(args):
    # Set the templates here
    # if args.template.find('har_mt') >= 0:
    #     args.save = 'har_mt'
    #     args.dataset = 'HAR'
    #     args.n_labeled = [288, 600,1200,'all']
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'

    # if args.template.find('har_sup_mt') >= 0:
    #     args.save = 'har_sup_mt'
    #     args.dataset = 'HAR'
    #     args.n_labeled = [288,600,1200]
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'
    #     args.n_labeled_per_batch = 100


    # if args.template.find('har_sp_mt') >= 0:
    #     args.save = 'har_sp_mt'
    #     args.dataset = 'HAR_SP'
    #     args.n_labeled = [288,600,1200,'all']
    #     args.input_dim = [17,16,9]
    #     args.cnn = 'har_sp'

    # if args.template.find('har_sp_sup_mt') >= 0:
    #     args.save = 'har_sp_sup_mt'
    #     args.dataset = 'HAR_SP'
    #     args.n_labeled = [288,600,1200]
    #     args.input_dim = [17,16,9]
    #     args.cnn = 'har_sp'
    #     args.n_labeled_per_batch = 100


    # dataset_detail 0: path of numpy file, 1, True for using ratio sampling
    if args.template.find('har_mt_bal') >= 0:
        args.save = 'har_mt_bal'
        args.dataset = 'HAR'
        args.n_labeled = [288, 600,1200,'all']
        args.input_dim = [9,64,1]
        args.cnn = 'har_conv2d'
        args.dataset_details = ['data/har/HAR_DATA_19_balanced.npz',False]

    if args.template.find('har_mt_un') >= 0:
        args.save = 'har_mt_un'
        args.dataset = 'HAR'
        args.n_labeled = [288, 600,1200,'all']
        args.input_dim = [9,64,1]
        args.cnn = 'har_conv2d'
        args.dataset_details = ['data/har/HAR_DATA_19_unbalanced25_per.npz',True]


    # if args.template.find('har_bal_sup_mt') >= 0:
    #     args.save = 'har_sup_mt'
    #     args.dataset = 'HAR'
    #     args.n_labeled = [288,600,1200]
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'
    #     args.n_labeled_per_batch = 100
    #     args.dataset_details = ['data/har/HAR_DATA_19_balanced.npz',False]

    # if args.template.find('har_un_sup_mt') >= 0:
    #     args.save = 'har_sup_mt'
    #     args.dataset = 'HAR'
    #     args.n_labeled = [288,600,1200]
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'
    #     args.n_labeled_per_batch = 100
    #     args.dataset_details = ['data/har/HAR_DATA_19_unbalanced25_per.npz',True]

    # if args.template.find('har_resize_mt') >= 0:
    #     args.save = 'har_resize_mt'
    #     args.dataset = 'HAR'
    #     args.n_labeled = [600,1200,'all']
    #     # args.n_labeled = [288]
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'
    #     args.resize = True

    # ema 0.99
    # if args.template.find('har_ema_mt') >= 0:
    #     args.save = 'har_ema_mt'
    #     args.dataset = 'HAR'
    #     # args.n_labeled = [300,600,1200,'all']
    #     args.n_labeled = [1200]
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'
    #     args.ema_decay_during_rampup = 0.99

    # if args.template.find('har_ema_sup_mt') >= 0:
    #     args.save = 'har_ema_sup_mt'
    #     args.dataset = 'HAR'
    #     # args.n_labeled = [300,600,1200]
    #     args.n_labeled = [1200]
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'
    #     args.n_labeled_per_batch = 100
    #     args.ema_decay_during_rampup = 0.99

    # if args.template.find('har_ema_sp_mt') >= 0:
    #     args.save = 'har_ema_sp_mt'
    #     args.dataset = 'HAR_SP'
    #     # args.n_labeled = [300,600,1200,'all']
    #     args.n_labeled = [1200]
    #     args.input_dim = [17,16,9]
    #     args.cnn = 'har_sp'
    #     args.ema_decay_during_rampup = 0.99

    # if args.template.find('har_ema_sp_sup_mt') >= 0:
    #     args.save = 'har_ema_sp_sup_mt'
    #     args.dataset = 'HAR_SP'
    #     # args.n_labeled = [300,600,1200]
    #     args.n_labeled = [1200]
    #     args.input_dim = [17,16,9]
    #     args.cnn = 'har_sp'
    #     args.n_labeled_per_batch = 100
    #     args.ema_decay_during_rampup = 0.99


    # if args.template.find('har_cls_wcons') >= 0:
    #     args.save = 'har_cls_wcons'
    #     args.dataset = 'HAR'
    #     args.n_classes = 24
    #     args.n_labeled = [1200]
    #     args.n_labeled_per_batch = 100
    #     args.input_dim = [9,64,1]
    #     args.flip_horizontally = False
    #     args.cnn = 'har_conv2d'
    #     args.training_length = 37500
    #     args.rampup_length = 10000
    #     args.ranpdown_length = 6250
    #     args.ema_decay_during_rampup = 0.99

    # if args.template.find('har_sp_cls_wcons') >= 0:
    #     args.save = 'har_sp_class'
    #     args.dataset = 'HAR_SP'
    #     args.n_classes = 24
    #     args.n_labeled = [1200]
    #     args.n_labeled_per_batch = 100
    #     args.input_dim = [17,16,9]
    #     args.flip_horizontally = False
    #     args.cnn = 'har_sp'
    #     args.training_length = 37500
    #     args.rampup_length = 10000
    #     args.ranpdown_length = 6250
    #     args.ema_decay_during_rampup = 0.99


    if args.template.find('har_conf_mt') >= 0:
        args.save = 'har_conf_mt'
        args.n_runs = 1
        args.dataset = 'HAR'
        args.n_labeled = ['all']
        args.input_dim = [9,64,1]
        args.cnn = 'har_conv2d'
        args.test_only = True
        args.ckp = 'har'

    # if args.template.find('har_confuse') >= 0:
    #     args.dataset = 'HAR'
    #     args.n_classes = 24
    #     args.n_labeled = ['all']
    #     args.input_dim = [9,64,1]
    #     args.cnn = 'har_conv2d'
    #     args.training_length = 37500
    #     args.rampup_length = 10000
    #     args.ranpdown_length = 6250
    #     args.ema_decay_during_rampup = 0.99

    # if args.template.find('har_sub_mt') >= 0:
    #     args.save = 'har_sub_mt'
    #     args.dataset = 'HAR'
    #     args.n_classes = 22
    #     args.n_labeled = [1100,'all']
    #     args.input_dim = [9,64,1]
    #     args.flip_horizontally = False
    #     args.cnn = 'har_conv2d'
    #     args.training_length = 37500
    #     args.rampup_length = 10000
    #     args.ranpdown_length = 6250
    #     args.ema_decay_during_rampup = 0.99
    #     args.dataset_detail = 'subject_all'

    # if args.template.find('har_sub_sup') >= 0:
    #     args.save = 'har_sub_sup'
    #     args.dataset = 'HAR'
    #     args.n_classes = 22
    #     args.n_labeled = [1100]
    #     args.n_labeled_per_batch = 100
    #     args.input_dim = [9,64,1]
    #     args.flip_horizontally = False
    #     args.cnn = 'har_conv2d'
    #     args.training_length = 37500
    #     args.rampup_length = 10000
    #     args.ranpdown_length = 6250
    #     args.ema_decay_during_rampup = 0.99
    #     args.dataset_detail = 'subject_all'

    # if args.template.find('har_sub_bal_mt') >= 0:
    #     args.save = 'har_sub_bal_mt'
    #     args.dataset = 'HAR'
    #     args.n_classes = 22
    #     args.n_labeled = [110,'all']
    #     args.input_dim = [9,64,1]
    #     args.flip_horizontally = False
    #     args.cnn = 'har_conv2d'
    #     args.training_length = 37500
    #     args.rampup_length = 10000
    #     args.ranpdown_length = 6250
    #     args.ema_decay_during_rampup = 0.99
    #     args.dataset_detail = 'subject_balanced'

    # if args.template.find('har_sub_bal_sup') >= 0:
    #     args.save = 'har_sub_bal_sup'
    #     args.dataset = 'HAR'
    #     args.n_classes = 22
    #     args.n_labeled = [110]
    #     args.n_labeled_per_batch = 100
    #     args.input_dim = [9,64,1]
    #     args.flip_horizontally = Fals11:00ame
    #     args.cnn = 'har_conv2d'11:00am
    #     args.training_length = 3750011:00am
    #     args.rampup_length = 1000011:00am
    #     args.ranpdown_length = 625011:00am
    #     args.ema_decay_during_rampup 11:00am= 0.99
    #     args.dataset_detail = 'subject_balanced'