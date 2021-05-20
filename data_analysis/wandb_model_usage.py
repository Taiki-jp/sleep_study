import wandb

wandb.init(project = "edl")

model = wandb.restore("model-best.h5",
                      run_path = "taiki/edl/28pgzug2")
