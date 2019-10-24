import pandas as pd
from pandas import read_msgpack
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='experiments')
parser.add_argument('--ratio', default=False, type=bool)
parser.add_argument('--exp', default='', type=str)
parser.add_argument('--best', default=False, type=bool) #best or last
args = parser.parse_args()


def msgpack2csv(msgdir,folder,n_runs=1):

    msgdir = msgdir.split('\n')
    dfmsgdir = pd.DataFrame({'dir':msgdir})
    dfmsgdir['dir'].replace('', np.nan, inplace=True)
    dfmsgdir = dfmsgdir.dropna()
    if args.ratio:
        dfmsgdir['ratio']=dfmsgdir['dir']
    dfmsgdir['cat']=dfmsgdir['dir']

    for i in range(dfmsgdir['dir'].shape[0]):
        dfmsgdir['cat'].iloc[i] = dfmsgdir['dir'].iloc[i].split('/')[-1]
        if args.ratio:
            dfmsgdir['ratio'].iloc[i] = dfmsgdir['dir'].iloc[i].split('/')[-3]

    # trmsg =  dfmsgdir.loc[dfmsgdir['cat'] == "training.msgpack"]
    # trmsg.loc[:,'ratio']=pd.to_numeric(trmsg['ratio'])
    # trmsg=trmsg.sort_values(['ratio'])

    # import pdb; pdb.set_trace()
    print(msgdir)
    # print(dfmsgdir['cat'])
    valmsg =  dfmsgdir.loc[dfmsgdir['cat'] == "validation.msgpack"]
    if args.ratio:
        valmsg.loc[:,'ratio']=pd.to_numeric(valmsg['ratio'])
        valmsg=valmsg.sort_values(['ratio'])

    valmsg['eval/error/ema']=''
    valmsg['eval/error/1']=''
    valmsg['eval/class_cost/ema']=''
    valmsg['eval/class_cost/1']=''
    valmsg['minepoch']=''

    for dir in valmsg['dir']:
        tmp = read_msgpack(dir)
        
        if args.best:
            minrcds = tmp.loc[tmp['eval/error/ema']==np.min(tmp['eval/error/ema'])]
        else:
            minrcds = tmp.loc[tmp.index==tmp.index[-1]]
        valmsg.loc[valmsg['dir']==dir,'minepoch'] = minrcds.index[0]
        valmsg.loc[valmsg['dir']==dir,'eval/error/ema']= minrcds.iloc[0]['eval/error/ema']
        valmsg.loc[valmsg['dir']==dir,'eval/error/1']= minrcds.iloc[0]['eval/error/1']
        valmsg.loc[valmsg['dir']==dir,'eval/class_cost/ema']= minrcds.iloc[0]['eval/class_cost/ema']
        valmsg.loc[valmsg['dir']==dir,'eval/class_cost/1']= minrcds.iloc[0]['eval/class_cost/1']

    def verify_runs(data,n_runs):
        for i in set(data):
            assert sum(data == i)==n_runs, print('Wrong number of runs for index {}'.format(i))

    if args.ratio:
        verify_runs(valmsg['ratio'].values,n_runs)
        print('Passed verifying')

        for i in sorted(set(valmsg['ratio'].values)):
            ema_avg = sum(valmsg.loc[valmsg['ratio']==i]['eval/error/ema'])/n_runs
            valmsg = valmsg.append({'eval/error/ema':ema_avg,'ratio':i},ignore_index=True)

    folder_name = folder.split('/')[-1]
    valmsg.to_csv('./results/csv/'+folder_name+"_vali.csv")
    print('File saved')

#put your directory in experiment
if __name__=="__main__":

    print('loading folers')
    root_path = "./results/"
    folders = os.listdir(root_path)
    assert args.exp != '','No experiments info was input'
    for i in range (len(folders)):
        folders[i] = os.path.join(root_path,folders[i])

    for folder in folders:
        if args.exp in folder:
            print(folder)
            msgdir = os.popen("find "+folder+"/ -name *.msgpack").read()
            msgpack2csv(msgdir,folder,4)
