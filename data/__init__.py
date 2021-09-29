# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>


from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset


class Data:
    def __init__(self, args):
        self.loader_train = []
        if not args.test_only:
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                trainset = (getattr(m, module_name)(args, name=d))

                self.loader_train.append(
                    dataloader.DataLoader(
                        trainset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        pin_memory=not args.cpu,
                        num_workers=args.n_threads,
                    )
                )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109','CBSD68','Rain100L','GOPRO_Large']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=args.test_batch_size,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
