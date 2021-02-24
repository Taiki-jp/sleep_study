import wandb

wandb.init(project = "FNN")

model = wandb.restore("model-best.h5",
                      run_path = "taiki/FNN/2lawwaa9")
