from collections import defaultdict

from tinygrad import Device
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from helpers.dataloader import DataLoader
import math
from tinygrad.nn.state import get_state_dict
from tinygrad import TinyJit


def log(step, max_steps, lr, metrics):
    metrics_print = " - ".join([f"{m}: {v[-1]:.3f}" for m, v in metrics.items()])
    print(f"Step {step + 1}/{max_steps} - LR:{lr:.4f} -", metrics_print, end="\r")


def cosine_annealing_lr(epoch, max_epochs, eta_min, eta_max):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / max_epochs))


def apply_weight_decay(model, weight_decay):
    for param in model.parameters():
        param.assign(param - param * weight_decay)


@TinyJit
def train(
    model,
    dl_train: DataLoader,
    lr: float,
    max_epochs: int,
    weight_decay: float = 1e-2,
    log_every: int = 10,
) -> defaultdict:
    print(f"Training on {Device.DEFAULT}.")

    metrics_tracker = defaultdict(list)
    Tensor.training = True
    # Filter out what would have been buffers in PyTorch (params you don't train)
    trainable_params = [param for name, param in get_state_dict(model).items() if "weight" in name]
    optimizer = Adam(trainable_params, lr=10 * lr)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}:")

        current_lr = cosine_annealing_lr(epoch, max_epochs, eta_min=lr, eta_max=10*lr)
        optimizer.lr = current_lr

        for step, (inputs, labels) in enumerate(dl_train):
            optimizer.zero_grad()

            logits = model(inputs)

            loss = logits.reshape(-1, logits.shape[-1]).sparse_categorical_crossentropy(labels.flatten())
            loss.backward()

            optimizer.step()

            # Debugging: Print gradients after optimizer step
            print("After optimizer step:")
            for name, param in get_state_dict(model).items():
                if param.requires_grad:
                    print(f"Parameter {name} grad: {param.grad}")

            # Manually apply weight decay
            apply_weight_decay(model, weight_decay)

            for param in model.parameters():
                print(f"Parameter after weight decay: {param[:5]}")  # Print first 5 elements

            metrics_tracker["train_loss"].append(loss.numpy().item())
            if step % log_every == 0 or step == len(dl_train) - 1:
                log(step, len(dl_train), optimizer.lr.numpy().item(), metrics_tracker)

    return metrics_tracker
    