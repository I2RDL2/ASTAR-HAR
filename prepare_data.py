import numpy as np
# path = 'data/har/stft_32_HAR_DATA_bigger_image.npz'
# path = 'data/har/HAR_DATA_balanced_diff_subject.npz'
path = 'data/har/HAR_DATA_same_subject_clean.npz'
# path = 'data/har/HAR_all_DATA_diff_subject.npz'
# output_path = './data/har/HAR_DATA_lok1.npz'
# output_path = './data/har/stft_8_HAR_DATA_new.npz'
# output_path1 = './data/har/stft_16_HAR_DATA_new.npz'
output_path = 'data/har/HAR_DATA_bal_diff_sub_lok.npz'
# output_path = 'data/har/HAR_DATA_all_diff_sub_lok.npz'
x = np.load(path)
# cifar_path='data/images/cifar/cifar10/cifar10_gcn_zca_v2.npz'

import pdb; pdb.set_trace()

train_x = x['trainx'] 
train_x = np.expand_dims(train_x, axis=-1)
# train_x = np.transpose(train_x,(0,3,2,1))
train_y = x['trainy']
test_x = x['testx']
test_x = np.expand_dims(test_x, axis=-1)
test_y = x['testy']


# train_x = x['trainx'] 
# train_x = np.transpose(train_x,(0,3,2,1))
# train_y = x['trainy']
# test_x = x['testx']
# test_x = np.transpose(test_x, (0,3,2,1))
# test_y = x['testy']

# for key in x.keys():
# 	print('shape of key {} is {}'.format(key, x[key].shape))
# import pdb; pdb.set_trace()
save_list = dict()
save_list.update(train_x=train_x, train_y=train_y,test_x=test_x, test_y=test_y)

for key in save_list.keys():
	print('shape of key {} is {}'.format(key, save_list[key].shape))	

np.savez(output_path,**save_list)

print('1')
