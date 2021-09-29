# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import utility
import SwinIR_image_utils as img_util
import torch
from tqdm import tqdm
import numpy as np
from decimal import *
import os

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        for _, d in enumerate(self.loader_train):
            for idx_scale, scale in enumerate(self.scale):
                print('scale: ', scale)
                d.dataset.set_scale(idx_scale)
                for batch, (lr, hr, _,) in enumerate(d):
                    lr, hr = self.prepare(lr, hr)
                    # model.module.set_scale(args.scale)
                    timer_data.hold()
                    timer_model.tic()

                    self.optimizer.zero_grad()
                    sr = self.model(lr, idx_scale)
                    loss = self.loss(sr, hr)
                    loss.backward()
                    if self.args.gclip > 0:
                        utils.clip_grad_value_(
                            self.model.parameters(),
                            self.args.gclip
                        )
                    self.optimizer.step()

                    timer_model.hold()

                    if (batch + 1) % self.args.print_every == 0:
                        self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                            (batch + 1) * self.args.batch_size,
                            len(d.dataset),
                            self.loss.display_loss(batch),
                            timer_model.release(),
                            timer_data.release()))

                    timer_data.tic()

        # if epoch%self.save_every ==0:
        #     self.model.save(self.save_path, epoch, False)
        self.loss.end_log(len(d))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        
    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range
                    )
                    #import pdb
                    #pdb.set_trace()
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    i = i+1
                    
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)

                if not self.args.test_only and best[0][idx_data, idx_scale] == self.ckp.log[-1, idx_data, idx_scale]: #save_best
                    print('saving mode...')
                    self.model.save(self.args.save_model_dir, best[1][idx_data, idx_scale]+1, is_best=True)
                    #torch.save(self.model.state_dict(), os.path.join(self.args.save_model_dir, 'x' + str(scale) + '_best.pt'))

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
