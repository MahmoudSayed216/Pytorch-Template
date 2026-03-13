import argparse
import sys
import yaml
import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from logger import Logger
from CheckpointsHandler import CheckpointsHandler
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from Model.simple_cnn import SimpleCNN
from torchvision import transforms
from dataset.CatDogDataset import CatDogDataset
from sklearn.model_selection import train_test_split
from dashboard.dashboard_reporter import DashboardReporter
from dashboard.config import CLASS_NAMES          # {"0": "Cat", "1": "Dog"}
import numpy as np


def load_configs() -> tuple[dict, dict]:
    
    # CONFIGS_YAML_FILE_PATH = os.path.join(os.path.dirname(__file__) ,"configs.yaml")
    CONFIGS_YAML_FILE_PATH = os.path.join(os.path.dirname(__file__) ,"configs.yaml")
    
    
    # with open(CONFIGS_YAML_FILE_PATH) as f:
        # training_configs = yaml.safe_load(f)
    with open(CONFIGS_YAML_FILE_PATH) as x:
        configs = yaml.safe_load(x)

    return configs

def parse_args() -> dict:

    parser = argparse.ArgumentParser("Model Training Configuration")

    #* Model
    parser.add_argument("--n_channels", type=int, default=None)

    #* Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)

    #* Checkpoint
    parser.add_argument("--continue-from-checkpoint", action="store_true")  # store_true is already False by default
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--checkpoint-type", type=str, choices=["best", "last"], default=None)

    #* Debug / Environment
    parser.add_argument("--use-debugger", action="store_true")  # False if not given
    parser.add_argument("--kaggle", action="store_true")         # False if not given
    parser.add_argument("--dataset-base-kaggle", type=str, default=None)
    parser.add_argument("--dataset-base-local", type=str, default=None)
    parser.add_argument("--output-dir-kaggle", type=str, default=None)
    parser.add_argument("--output-dir-local", type=str, default=None)

    #* Device
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)

    args = parser.parse_args()


    return args.__dict__

def override_configs(configs: dict, args: dict):

    if args["n_channels"] is not None:
        configs["model"]["n_channels"] = args["n_channels"]

    if args["lr"] is not None:
        configs["training"]["learning_rate"] = args["lr"]
    if args["patience"] is not None:
        configs["training"]["patience"] = args["patience"]
    if args["epochs"] is not None:
        configs["training"]["epochs"] = args["epochs"]
    if args["batch_size"] is not None:
        configs["training"]["batch_size"] = args["batch_size"]
    if args["start_epoch"] is not None:
        configs["training"]["start_epoch"] = args["start_epoch"]
    if args["save_every"] is not None:
        configs["training"]["save_every"] = args["save_every"]

    configs["checkpoint"]["continue"] = args["continue_from_checkpoint"]
    
    if args["checkpoint_id"] is not None:
        configs["checkpoint"]["id"] = args["checkpoint_id"]
    if args["checkpoint_type"] is not None:
        configs["checkpoint"]["type"] = args["checkpoint_type"]

    configs["use_debugger"] = args["use_debugger"] 
    configs["kaggle"] = args["kaggle"]

    if args["dataset_base_kaggle"] is not None:
        configs["environment"]["dataset_base"]["kaggle"] = args["dataset_base_kaggle"]
    if args["dataset_base_local"] is not None:
        configs["environment"]["dataset_base"]["local"] = args["dataset_base_local"]
    
    if args["output_dir_kaggle"] is not None:
        configs["environment"]["output_dir"]["kaggle"] = args["output_dir_kaggle"]
    if args["output_dir_local"] is not None:
        configs["environment"]["output_dir"]["local"] = args["output_dir_local"]
    
    if args["device"] is not None:
        configs["device"] = args["device"]

def resolve_paths(shared_configs: dict) -> tuple[str, str]:
    if shared_configs["environment"]["kaggle"]:
        dataset_path = shared_configs["environment"]["dataset_base"]["kaggle"]
        output_path = shared_configs["environment"]["output_dir"]["kaggle"]
    else:
        dataset_path = shared_configs["environment"]["dataset_base"]["local"]
        output_path = shared_configs["environment"]["output_dir"]["local"]
    # print()
    return dataset_path, output_path

def create_training_environment(output_relative_path: str) -> str:
    BASE_DIR = Path(__file__).resolve().parent #! TRY os.path.dirname INSTEAD
    output_path_base = os.path.join(BASE_DIR, output_relative_path)
    print("OUTPUT PATH: ", output_path_base)
    if not os.path.exists(output_path_base):
        os.mkdir(os.path.join(output_path_base))
    
    session_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    session_path = os.path.join(output_path_base, session_id)
    os.mkdir(session_path)


    logs_folder = os.path.join(session_path, "logs")
    os.mkdir(path=logs_folder)
    weights_folder = os.path.join(session_path, "weights")
    os.mkdir(path=weights_folder)

    return session_path, session_id

def log_configs(logger, session_path, configs):
    key_val_pairs = []

    def traverse_dict(d, parent_key=""):
        for key, value in d.items():
            full_key = f"{parent_key}-{key}" if parent_key else str(key)
            if isinstance(value, dict):
                traverse_dict(value, full_key)
            else:
                key_val_pairs.append(f"{full_key}: {value}")

    traverse_dict(configs)
    output = "\n".join(key_val_pairs)
    output = "\nconfigs:\n" + output
    logger.log(output)
    key_val_pairs.clear()

    logger.log("session path: ", session_path, end="\n\n")


def create_data_loaders(dataset_path: str, configs: dict) -> tuple[DataLoader, DataLoader]:

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),              # Resize to 224x224
        transforms.RandomHorizontalFlip(p=0.5),    # Randomly flip images
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Slight color jitter
        transforms.ToTensor(),                      # Convert to tensor
        transforms.Normalize(                       # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
        
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    
    labels_map = {"Cat": 0, "Dog": 1}

    train_samples = []
    test_samples = []

    for label_name, label_idx in labels_map.items():
        folder = os.path.join(dataset_path, label_name)
        files = [os.path.join(folder, f) for f in os.listdir(folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        
        train_samples += [(f, label_idx) for f in train_files]
        test_samples += [(f, label_idx) for f in test_files]

    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

    train_ds = CatDogDataset(train_samples ,transform=train_transforms)
    test_ds = CatDogDataset(test_samples, transform=test_transform)

    device = configs["device"]
    
    train_loader = DataLoader(dataset=train_ds, num_workers=5, shuffle=True, batch_size=configs["training"]["batch_size"], pin_memory=(device == "cuda"))
    test_loader = DataLoader(dataset=test_ds, num_workers=5 , batch_size=1, pin_memory=(configs["device"] == "cuda"))

    
    return train_loader, test_loader




#TODO: implement this function
#* Move to test.py
import torch

def compute_test_metrics(model, loss_fn, device, test_loader, train_loader=None):
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            output = model(imgs)
            loss = loss_fn(output, labels)
            test_loss += loss.item() * imgs.size(0)  # multiply by batch size

            pred = (torch.sigmoid(output) > 0.5).float()
            correct_test += (pred == labels).sum().item()
            total_test += imgs.size(0)

    test_loss /= total_test
    test_accuracy = correct_test / total_test

    train_accuracy = None
    if train_loader is not None:
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                output = model(imgs)
                pred = (torch.sigmoid(output) > 0.5).float()
                correct_train += (pred == labels).sum().item()
                total_train += imgs.size(0)
        train_accuracy = correct_train / total_train

    return test_loss, test_accuracy, train_accuracy

def load_checkpoint(session_path: str, state_type, device, model_name, session_id, checkpoint_id):
    session_path = session_path.replace(session_id, checkpoint_id)
    path_to_ckpt = os.path.join(os.getcwd(), f'src/{model_name}',session_path, 'weights', f'{state_type}.pth')
    ckpt = torch.load(path_to_ckpt, map_location=device)
    return ckpt


def _send_samples(model, test_loader, loss_fn, device, n=10):
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
                img_np = ((img_np - img_np.min()) / (np.ptp(img_np) + 1e-8) * 255).astype(np.uint8)
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
 

    
#TODO: implement this function
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
        _send_samples(model, test_loader, loss_fn, DEVICE, reporter, n=10)

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




def main():
    configs = load_configs()
    args = parse_args()
    
    if len(sys.argv) > 1:
        override_configs(configs, args)

    dataset_path, output_path = resolve_paths(configs)
    
    session_path, session_id = create_training_environment(output_path)

    logger = Logger(debug_mode=configs["environment"]["debugger_active"], logs_folder_path=os.path.join(session_path, "logs"))



    #TODO: LOG ALL CONFIGS AND SESSION PATH BEFORE STARTING
    log_configs(logger=logger, session_path=session_path, configs=configs)
    # print(os.listdir(session_path))
    train_loader, test_loader = create_data_loaders(dataset_path, configs)
    
    SERVER_URL = configs["environment"]["SERVER_URL"]
    reporter = DashboardReporter(server_url=SERVER_URL)


    train(train_loader = train_loader,test_loader =  test_loader, configs=configs, session_path=session_path, session_id=session_id,logger=logger, reporter=reporter)


if __name__ == "__main__":

    main()