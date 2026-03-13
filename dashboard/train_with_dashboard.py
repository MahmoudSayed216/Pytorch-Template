# ─────────────────────────────────────────────────────────────────────────────
#  train_with_dashboard.py  –  runs on Kaggle
#
#  Copy dashboard_reporter.py to Kaggle alongside this file.
#  Set SERVER_URL to the ngrok URL printed by app.py on your local machine.
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCEWithLogitsLoss

from dashboard_reporter import DashboardReporter
from config import CLASS_NAMES          # {"0": "Cat", "1": "Dog"}

# ── Paste the ngrok URL printed by app.py here ─────────────────────────────────

# ── Helper: build sample payload after each epoch ─────────────────────────────

def _send_samples(model, test_loader, loss_fn, device, reporter, n=10):
    """Pick n random test images, run inference, send to dashboard."""
    model.eval()
    all_imgs, all_true, all_pred, all_conf = [], [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)

            for img_t, true_l, pred_l, conf in zip(imgs.cpu(), labels.numpy(), preds, probs):
                # Convert tensor → numpy HWC uint8
                img_np = img_t.permute(1, 2, 0).numpy()
                img_np = ((img_np - img_np.min()) / (img_np.ptp() + 1e-8) * 255).astype(np.uint8)
                all_imgs.append(img_np)
                all_true.append(CLASS_NAMES[int(true_l)])
                all_pred.append(CLASS_NAMES[int(pred_l)])
                all_conf.append(float(conf))

            if len(all_imgs) >= n:
                break

    # Shuffle and pick n random ones
    idx = np.random.choice(len(all_imgs), size=min(n, len(all_imgs)), replace=False)
    reporter.log_samples(
        images=[all_imgs[i] for i in idx],
        true_labels=[all_true[i] for i in idx],
        pred_labels=[all_pred[i] for i in idx],
        confidences=[all_conf[i] for i in idx],
    )
    model.train()


# ── Training loop ──────────────────────────────────────────────────────────────

def train(train_loader: DataLoader, test_loader: DataLoader, configs: dict,
          session_path: str, session_id: str, logger, reporter):

    EPOCHS             = configs["training"]["epochs"]
    LEARNING_RATE      = configs["training"]["learning_rate"]
    SAVE_EVERY         = configs["training"]["save_every"]
    START_EPOCH        = 0
    DEVICE             = configs["device"]
    MODEL_NAME         = configs["model"]["name"]
    LR_REDUCTION_FACTOR = configs["training"]["gamma"]
    LR_REDUCE_AFTER    = configs["training"]["reduce_lr_after"]
    weights_saving_path = os.path.join(session_path, "weights")

    cp_handler = CheckpointsHandler(save_every=SAVE_EVERY, increasing_metric=True, output_path=weights_saving_path)
    model      = SimpleCNN().to(DEVICE)
    optim      = Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler  = StepLR(optimizer=optim, step_size=LR_REDUCE_AFTER, gamma=LR_REDUCTION_FACTOR)
    loss_fn    = BCEWithLogitsLoss()
    logger.debug(string=f"Session path: {session_path}")

    if configs["checkpoint"]["continue"]:
        checkpoint_type = configs["checkpoint"]["type"]
        checkpoint = load_checkpoint(session_path, checkpoint_type, DEVICE,
                                     model_name=configs["model"]["name"],
                                     session_id=session_id,
                                     checkpoint_id=configs["checkpoint"]["id"])
        model.load_state_dict(checkpoint["model"])
        cp_handler.previous_best_value = checkpoint["score"]
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["sched"])
        START_EPOCH = checkpoint["epoch"] + 1

    # ── Tell dashboard we're starting fresh ────────────────────────────────────
    reporter.reset()

    logger.log(f"Training {MODEL_NAME} starting for {EPOCHS-START_EPOCH} epochs, "
               f"Learning rate = {LEARNING_RATE}, with Adam optimizer")

    for epoch in range(START_EPOCH, EPOCHS + 1):
        model.train()
        logger.log(f"Epoch: {epoch}")
        epoch_cummulative_loss = 0
        steps = 0

        for i, (input_imgs, labels) in enumerate(train_loader):
            optim.zero_grad()
            input_imgs = input_imgs.to(DEVICE)
            output     = model(input_imgs)
            labels     = labels.to(DEVICE).float().unsqueeze(1)
            loss       = loss_fn(output, labels)
            loss.backward()
            optim.step()

            epoch_cummulative_loss += loss.item()
            steps += 1

            # ── [DASHBOARD] send per-step train loss ───────────────────────────
            reporter.log_step(loss=loss.item())

        scheduler.step()
        avg_train_mse = epoch_cummulative_loss / steps

        # (unchanged) compute metrics
        test_loss, test_accuracy, train_accuracy = compute_test_metrics(
            model, loss_fn, DEVICE, test_loader, train_loader)

        logger.log(f"Average Train Loss = {avg_train_mse:.12f}")
        logger.log(f"Train Accuracy: {train_accuracy}")
        logger.log(f"Test Loss = {test_loss:.12f}")
        logger.log(f"Test Accuracy: {test_accuracy}")

        # ── [DASHBOARD] send per-epoch metrics ────────────────────────────────
        reporter.log_epoch(
            epoch=epoch,
            avg_train_loss=avg_train_mse,
            test_loss=test_loss,
            train_acc=train_accuracy,
            test_acc=test_accuracy,
        )

        # ── [DASHBOARD] send sample predictions ──────────────────────────────
        _send_samples(model, test_loader, loss_fn, DEVICE, n=10)

        # (unchanged) checkpointing logic
        if cp_handler.check_save_every(epoch):
            logger.checkpoint(f"{SAVE_EVERY} epochs have passed, saving data in last.pth")
            cp_handler.save_model({
                'model': model.state_dict(), 'optim': optim.state_dict(),
                'sched': scheduler.state_dict(), 'test_accuracy': test_accuracy,
                'test_loss': test_loss, 'epoch': epoch, 'preds': ["samples"],
            }, save_type='last')

        if cp_handler.metric_has_improved(test_accuracy):
            logger.checkpoint(f"metric has improved, saving data in best.pth")
            cp_handler.save_model({
                'model': model.state_dict(), 'optim': optim.state_dict(),
                'sched': scheduler.state_dict(), 'test_accuracy': test_accuracy,
                'test_loss': test_loss, 'epoch': epoch, 'preds': ["samples"],
            }, save_type='best')
