import os, time, argparse, torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, "/kaggle/working/vietocr")  # path to your cloned repo

from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer  # we reuse its model/loss/datasets

def setup(local_rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # Kaggle-friendly NCCL flags
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE",  "1")
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--cfg", help="YAML config path")
    g.add_argument("--cfg-name", default="vgg_seq2seq", help="built-in config")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--val-ann",   required=True)
    ap.add_argument("--out", default="/kaggle/working/weights_ddp")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs",     type=int, default=128)
    ap.add_argument("--eval-bs",type=int, default=256)
    ap.add_argument("--workers",type=int, default=4)
    ap.add_argument("--lr",     type=float, default=3e-4)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    setup(local_rank, world_size)

    # ---- build VietOCR bits via Trainer ----
    cfg = Cfg.load_config_from_file(args.cfg) if args.cfg else Cfg.load_config_from_name(args.cfg_name)
    cfg['dataset'] |= {
        'data_root': args.data_root,
        'train_annotation': args.train_ann,
        'valid_annotation': args.val_ann,
        'batch_size': args.bs,
        'eval_batch_size': args.eval_bs
    }
    cfg['optimizer']['lr']   = args.lr
    cfg['trainer']['epochs'] = args.epochs

    t = Trainer(cfg, pretrained=False)
    model, criterion, optimizer = t.model, t.loss, t.optimizer
    train_ds = t.train_gen.dataset
    val_ds   = t.val_gen.dataset if t.val_gen is not None else None
    collate  = t.train_gen.collate_fn

    # ---- DDP + DataLoaders ----
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True)
    train_loader  = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True, collate_fn=collate)

    scaler = torch.cuda.amp.GradScaler(True)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    # ---- train ----
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running, t0 = 0.0, time.time()
        for batch in train_loader:
            imgs  = batch['image'].cuda(local_rank, non_blocking=True)
            labels= batch['label'].cuda(local_rank, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        if local_rank == 0:
            ckpt = os.path.join(args.out, f"epoch{epoch+1:03d}.pt")
            torch.save({"model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch+1}, ckpt)
            print(f"[epoch {epoch+1}/{args.epochs}] "
                  f"loss={running/max(1,len(train_loader)):.4f} "
                  f"time={time.time()-t0:.1f}s | saved {ckpt}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
