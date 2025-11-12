# vietocr/model/trainer_accel.py
import os, math, time, torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Optional
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

@dataclass
class TrainArgs:
    cfg_file: Optional[str] = None
    cfg_name: Optional[str] = None       # e.g. "vgg_seq2seq"
    data_root: Optional[str] = None
    train_ann: Optional[str] = None
    val_ann: Optional[str] = None
    out_dir: str = "runs/seq2seq_accel"
    epochs: Optional[int] = None
    lr: Optional[float] = None
    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    num_workers: int = 4
    mixed_precision: str = "fp16"
    accumulate_steps: int = 1
    seed: int = 42

def _build_from_trainer(cfg):
    t = Trainer(cfg, pretrained=False)
    return (
        t.model, t.loss, t.optimizer,
        t.train_gen.dataset,
        (t.val_gen.dataset if t.val_gen is not None else None),
        t.train_gen.collate_fn
    )

def train_main(args: TrainArgs):
    # 1) load & override cfg
    if args.cfg_file: cfg = Cfg.load_config_from_file(args.cfg_file)
    else:             cfg = Cfg.load_config_from_name(args.cfg_name or "vgg_seq2seq")
    if args.data_root:  cfg['dataset']['data_root'] = args.data_root
    if args.train_ann:  cfg['dataset']['train_annotation'] = args.train_ann
    if args.val_ann:    cfg['dataset']['valid_annotation'] = args.val_ann
    if args.batch_size: cfg['dataset']['batch_size'] = args.batch_size
    if args.eval_batch_size:
        cfg['dataset']['eval_batch_size'] = args.eval_batch_size
    if args.lr:         cfg['optimizer']['lr'] = args.lr
    if args.epochs:     cfg['trainer']['epochs'] = args.epochs
    epochs = cfg['trainer']['epochs']
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    model, criterion, optimizer, train_ds, val_ds, collate = _build_from_trainer(cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg['dataset']['batch_size'], shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=cfg['dataset'].get('eval_batch_size', cfg['dataset']['batch_size']),
            shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate
        )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            imgs, labels = batch['image'], batch['label']
            with accelerator.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels) / args.accumulate_steps
            accelerator.backward(loss)
            if (i + 1) % args.accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()

        if accelerator.is_main_process:
            ckpt = os.path.join(args.out_dir, f"epoch{epoch+1:03d}.pt")
            torch.save(
                {"model": accelerator.unwrap_model(model).state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1},
                ckpt
            )
            accelerator.print(f"[epoch {epoch+1}/{epochs}] loss={running/max(1,len(train_loader)):.4f} "
                              f"time={time.time()-t0:.1f}s | saved {ckpt}")
