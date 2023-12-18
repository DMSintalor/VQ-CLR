import torch


def load_optimizer(args, model):
    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    return optimizer, scheduler
