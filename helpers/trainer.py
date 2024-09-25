from collections import defaultdict
from tinygrad import Device
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

def log(step, max_steps, lr, metrics):
    metrics_print = " - ".join([f"{m}: {v[-1]:.3f}" for m, v in metrics.items()])
    print(f"Step {step + 1}/{max_steps} - LR:{lr:.4f} -", metrics_print, end="\r")


def train(
    model,
    dl_train,
    lr: float,
    max_epochs: int,
    weight_decay: float = 1e-2,
    log_every: int = 10,
) -> defaultdict:
    print("Training on {Device.default}.")

    metrics_tracker = defaultdict(list)
    Tensor.training = True
    optimizer = AdamW(get_parameters(model), lr=10 * lr, weight_decay=weight_decay)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}:")
        for step, (inputs, labels) in enumerate(dl_train):
            optimizer.zero_grad()

            logits = model(inputs)

            loss = logits.reshape(-1, logits.shape[-1]).sparse_categorical_crossentropy(labels.flatten())
            loss.backward()
                
            optimizer.step()

            metrics_tracker["train_loss"].append(loss.numpy().item())
            if step % log_every == 0 or step == len(dl_train) - 1:
                log(step, len(dl_train), optimizer.lr.numpy().item(), metrics_tracker)

    return metrics_tracker
    