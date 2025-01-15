from data_provider.data_loader import Dataset_hour
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'DIA_DOG_5min': Dataset_hour,
    'DDM_DXD_5min': Dataset_hour,
    'UDOW_SDOW_5min': Dataset_hour,
    'QQQ_PSQ_5min': Dataset_hour,
    'QLD_QID_5min': Dataset_hour,
    'TQQQ_SQQQ_5min': Dataset_hour,
    'SPY_SH_5min': Dataset_hour,
    'SSO_SDS_5min': Dataset_hour,
    'UPRO_SPXU_5min': Dataset_hour

}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            timeenc=timeenc,
            freq=freq
        )
    print(flag, len(data_set))
    
    data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    
    return data_set, data_loader