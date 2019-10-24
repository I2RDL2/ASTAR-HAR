# python3 experiments/har/har.py --template har_mt

# python3 experiments/har/har.py --template har_class

# python3 experiments/har/har.py --template har_sp_mt

# python3 experiments/har/har.py --template har_sp_class


# python3 experiments/har/har.py --template har_cls_wcons

# python3 experiments/har/har.py --template har_sp_cls_wcons



# python3 experiments/har/har.py --template har_confuse --test_only --ckp results/har_mt_1200/2000/transient/checkpoint-37501

# python3 experiments/har/har.py --template har_confuse --test_only --ckp results/har_cls_wcons_1200/2000/transient/checkpoint-37501

# python3 experiments/har/har.py --template har_cls_wcons --test_only --ckp results/har_cls_wcons_1200/transient

# python3 experiments/har/har.py --template har_cls_wcons --test_only --ckp results/har_cls_wcons_1200/transient

# python3 experiments/har/har.py --template har_sub_mt

# python3 experiments/har/har.py --template har_sub_sup

# python3 experiments/har/har.py --template har_sub_bal_mt

# python3 experiments/har/har.py --template har_sub_bal_sup


# python3 experiments/har/har.py --template har_mt

# python3 experiments/har/har.py --template har_sup_mt

# python3 experiments/har/har.py --template har_sp_mt

# python3 experiments/har/har.py --template har_sp_sup_mt

# python3 experiments/har/har.py --template har_ema_mt

# python3 experiments/har/har.py --template har_ema_sup_mt

# python3 experiments/har/har.py --template har_ema_sp_mt

# python3 experiments/har/har.py --template har_ema_sp_sup_mt


# python3 experiments/har/har.py --template har_conf_mt

# python3 experiments/har/har.py --template har_resize_mt



#  experiments for har data with mean teacher
python3 experiments/har/har.py --template har_mt
#  experiments for har data in supervised manner
python3 experiments/har/har.py --template har_sup_mt
# for compting confuse matrix
python3 experiments/har/har.py --template har_conf_mt