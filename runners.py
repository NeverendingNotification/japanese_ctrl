
import os
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from torch.optim.lr_scheduler import LambdaLR

def save_check_points(log_dir, model, optimizer, step, epoch):
    model_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(model_dir, exist_ok=True)
    check_point = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(check_point, os.path.join(model_dir, "checkpoint_{:08d}.pth".format(step)))


def train(model, optimizer, num_epochs,
        train_loader, log_dir, valid_loader=None, device="cpu", writer=None, scheduler=None,
        ignore_index=-100, log_period=20, tester=None, iter_count=1, start_epoch=1, check_point_path=None):
    if check_point_path is not None:
        print("loading checkpoint from ", check_point_path)
        check_point_path = torch.load(check_point_path)
        model.load_state_dict(check_point_path["model"])
        optimizer.load_state_dict(check_point_path["optimizer"])
        iter_count = check_point_path["step"]
        start_epoch = check_point_path["epoch"] + 1

    criteria = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss_df = []
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        losses = []
        with tqdm(train_loader) as prg:
            for x in prg:
                input_ids = x.to(device)
                lm_logits, _ = model(input_ids)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = criteria(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) 
                        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if len(losses) % log_period == 0:
                    prg.set_description("Epoch {}/{}  Iter {} : loss {:.4f} ".format(epoch, num_epochs, iter_count, np.mean(losses)))
                iter_count += 1
                if scheduler is not None:
                    scheduler.step()

        tr_loss = np.mean(losses)
        print("Epoch {}/{} : train loss {:.4f}".format(
            epoch, num_epochs, tr_loss))
        loss_df.append((epoch, tr_loss))
        save_check_points(log_dir, model, optimizer, iter_count, epoch)

        if tester is not None:
            tester(train_loader.dataset, model, device, "test_{:04d}.csv".format(epoch))
        if writer is not None:
            writer.add_scalars("loss" ,{ 
                "train_loss":tr_loss,
                "valid_loss": va_loss},
                global_step=epoch)

    loss_df = pd.DataFrame(loss_df, columns=["epoch", "train_loss"]).set_index("epoch")
    return model, loss_df

class Trainer:
    def __init__(self, datasets, models, opt_type="adagrad", lr_factor=1.0,
                num_epochs=50, log_dir="./logs", lr_schedule="iter_warmup",
                iter_warmup_period=1000, test_file=None,
                 **other_params):
        self.datasets = datasets
        self.ignore_index = datasets["train"].dataset.pad_code

        self.models = models
        if opt_type == "adam":
            self.optimizer_class = torch.optim.Adam
            self.base_lr = 0.001 * lr_factor
            self.optimizer = torch.optim.Adam(self.models["model"].parameters(), lr=self.base_lr)
        elif opt_type == "adagrad":
            self.base_lr = 0.01 * lr_factor
            self.optimizer = torch.optim.Adagrad(self.models["model"].parameters(), lr=self.base_lr)
        elif opt_type == "sgd":
            self.base_lr = 0.1 * lr_factor
            self.optimizer = torch.optim.SGD(self.models["model"].parameters(), lr=self.base_lr, weight_decay=1e-4, momentum=0.9,
                                    nesterov=True)
        else:
            raise NotImplementedError()
        print("optimizer : ", self.optimizer)
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
#        self.summary = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))
        self.summary = None
        self.lr_schedule = lr_schedule
        if test_file is not None:
            self.tester = Tester(test_file, log_dir)
        else:
            self.tester = None
        self.other_params = other_params
        self.iter_warmup_period = iter_warmup_period

    def __call__(self):
        model = self.models["model"]
        optimizer = self.optimizer
        if self.lr_schedule is not None:
            if self.lr_schedule == "step":
                decay_func = get_decay_func(self.num_epochs)
                scheduler = LambdaLR(optimizer, lr_lambda = decay_func)
            elif self.lr_schedule == "iter_warmup":
                def warmup_func(step):
                    return min(1.0, step/self.iter_warmup_period)
                scheduler = LambdaLR(optimizer, lr_lambda = warmup_func)
        else:
            scheduler = None

        model, log_df = train(model, optimizer, self.num_epochs, self.datasets["train"],  self.log_dir,
            valid_loader=self.datasets.get("valid", None),
            device=self.models["device"], writer=self.summary, scheduler=scheduler,
            ignore_index=self.ignore_index, tester=self.tester, **self.other_params)
        log_df.to_csv(os.path.join(self.log_dir, "log.csv"))
#        self.summary.close()
        torch.save(model.state_dict(), os.path.join(self.log_dir, "model.pth"))

class Tester:
    def __init__(self, test_csv, log_dir, test_col="text", max_length=50):
        self.texts = pd.read_csv(test_csv)[test_col].values
        self.log_dir = os.path.join(log_dir, "test")
        os.makedirs(self.log_dir, exist_ok=True)
        self.max_length = max_length

    def __call__(self, dataset, model, device, log_name):
        sp_test = dataset.sp
        code_ids = dataset.codes_ids
        ctrl_codes = dataset.ctrl_codes
        rows = []
        for prompt_text in self.texts:
            outs = [prompt_text]
            for code_id in code_ids:
                encoded = torch.Tensor( [[code_id] + sp_test.encode_as_ids(prompt_text)]).long().to(device)
                generated = model.generate(encoded, max_length=self.max_length)
                gen_list=list(map(int, generated.cpu().numpy()[0]))
                out = sp_test.decode_ids(gen_list[1:])
                outs.append(out)
            rows.append(outs)
        res_df = pd.DataFrame(rows, columns=["序文"] + ctrl_codes)
        res_df.to_csv(os.path.join(self.log_dir, log_name))


def get_runners(params, datasets, models):
    run_params = params["run_params"]
    runners = []
    for key, param in run_params.items():
        if key == "train":
            runners.append(Trainer(datasets, models,
                                log_dir=params["main_params"]["log_dir"],
                                **param))
        else:
            raise NotImplementedError()
    return runners