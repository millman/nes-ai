from pathlib import Path
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
import action_distance_cfm as adc

def main():
    cfg = adc.TrainCfg(
        data_root=Path("data.image_distance.train_levels_1_2"),
        out_dir=Path("prof.tmp"),
        epochs=1,
        batch_size=32,
        viz_every=0,
        log_every=1,
    )
    cfg.device = "mps"

    device = adc.pick_device(cfg.device)
    adc.set_seed(cfg.seed)

    enc = adc.Encoder(cfg.z_dim, pretrained=cfg.encoder_pretrained).to(device)
    dec = adc.Decoder(cfg.z_dim).to(device)
    vf  = adc.VectorField(cfg.z_dim).to(device)

    optim_kwargs = {"lr": cfg.lr, "weight_decay": 1e-4}
    if cfg.use_foreach_optim:
        optim_kwargs["foreach"] = True
    try:
        opt = torch.optim.AdamW([
            {"params": enc.parameters(), "lr": cfg.lr},
            {"params": dec.parameters(), "lr": cfg.lr},
            {"params": vf.parameters(),  "lr": cfg.lr * cfg.vf_lr_mult},
        ], **optim_kwargs)
    except TypeError:
        if optim_kwargs.pop("foreach", None) is not None:
            opt = torch.optim.AdamW([
                {"params": enc.parameters(), "lr": cfg.lr},
                {"params": dec.parameters(), "lr": cfg.lr},
                {"params": vf.parameters(),  "lr": cfg.lr * cfg.vf_lr_mult},
            ], **optim_kwargs)
        else:
            raise

    ds = adc.PairFromTrajDataset(cfg.data_root, "train", 0.95, cfg.seed,
                                cfg.max_step_gap, cfg.allow_cross_traj, cfg.p_cross_traj)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers, drop_last=True)

    sched = schedule(wait=0, warmup=2, active=6, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler("profiler_logs"),
    ) as prof:
        data_it = iter(dl)
        for step in range(4):  # profile a few steps
            A, B, _, _ = next(data_it)
            A = adc.to_float01(A, device, non_blocking=False)
            B = adc.to_float01(B, device, non_blocking=False)

            zA = enc(A)
            zB = enc(B)
            t = torch.rand(A.shape[0], device=device)

            zA_d, zB_d = zA.detach(), zB.detach()
            zt = (1.0 - t[:, None]) * zA_d + t[:, None] * zB_d
            u = zB_d - zA_d
            v = vf(zt, t, zA_d, zB_d)

            cos_term = 1.0 - torch.nn.functional.cosine_similarity(v, u, dim=1).mean()
            mag_term = (torch.linalg.norm(v - u, dim=1) /
                        (torch.linalg.norm(u, dim=1) + 1e-6)).mean()
            loss_cfm = cos_term + mag_term

            x_pair = dec(torch.cat([zA, zB], dim=0))
            xA, xB = x_pair.chunk(2, dim=0)
            loss_rec = torch.nn.functional.l1_loss(xA, A) + torch.nn.functional.l1_loss(xB, B)

            loss = cfg.lambda_cfm * loss_cfm + cfg.lambda_rec * loss_rec
            opt.zero_grad()
            loss.backward()
            opt.step()

            prof.step()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

if __name__ == "__main__":
    main()
