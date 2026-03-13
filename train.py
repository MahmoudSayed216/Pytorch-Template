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
from .logger import Logger
from .CheckpointsHandler import CheckpointsHandler
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from Model.simple_cnn import SimpleCNN

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

def log_all_configs(logger, session_path, training_configs, shared_configs):
    key_val_pairs =  []
    def traverse_dict(d, parent_key=""):
        for key, value in d.items():
            full_key = f"{parent_key}-{key}" if parent_key else str(key)

            if isinstance(value, dict):
                traverse_dict(value, full_key)
            else:
                key_val_pairs.append(f"{full_key}: {value}")

    traverse_dict(training_configs)
    output = "\n".join(key_val_pairs)
    output = "\ntrain configs: "+"\n" + output
    logger.log(output)
    key_val_pairs.clear()

    traverse_dict(shared_configs)
    output = "\n".join(key_val_pairs)
    output = "\nshared configs: "+"\n" + output
    logger.log(output)
    key_val_pairs.clear()

    logger.log("session path: ", session_path, end="\n\n")

    # print(output)


#TODO: implement this function
def create_data_loaders(dataset_path: str, training_configs: dict, configs: dict) -> tuple[DataLoader, DataLoader]:
    # train_ds = GoProDataset(dataset_path, split="train", crops=True,transforms=augment_patch)
    # test_ds = GoProDataset(dataset_path, split="test", crops=False, transforms=None)

    # device = configs["device"]
    
    # train_loader = DataLoader(dataset=train_ds, num_workers=5, shuffle=True, batch_size=training_configs["training"]["batch_size"], pin_memory=(device == "cuda"))
    # test_loader = DataLoader(dataset=test_ds, num_workers=5 , batch_size=1, pin_memory=(configs["device"] == "cuda"))

    
    # return train_loader, test_loader
    pass




#TODO: implement this function
#* Move to test.py
def compute_test_metrics(model, loss_fn, device, test_loader, mx) -> tuple[float, float]:
    pass

def load_checkpoint(session_path: str, state_type, device, model_name, session_id, checkpoint_id):
    session_path = session_path.replace(session_id, checkpoint_id)
    path_to_ckpt = os.path.join(os.getcwd(), f'src/{model_name}',session_path, 'weights', f'{state_type}.pth')
    ckpt = torch.load(path_to_ckpt, map_location=device)
    return ckpt

    
#TODO: implement this function
def train(train_loader: DataLoader, test_loader: DataLoader, training_configs: dict, shared_configs: dict, session_path: str, session_id: str, logger: Logger):
    EPOCHS = training_configs["training"]["epochs"]
    LEARNING_RATE = training_configs["training"]["learning_rate"]
    SAVE_EVERY = training_configs["training"]["save_every"]
    START_EPOCH = 0
    DEVICE = shared_configs["device"]
    MODEL_NAME = training_configs["model"]["name"]
    LR_REDUCTION_FACTOR = training_configs["training"]["gamma"]
    LR_REDUCE_AFTER = training_configs["training"]["reduce_lr_after"]
    weights_saving_path = os.path.join(session_path, "weights")


    cp_handler = CheckpointsHandler(save_every=SAVE_EVERY, increasing_metric=True, output_path=weights_saving_path)
    model = SimpleCNN(train_configs=training_configs, shared_configs=shared_configs).to(DEVICE)
    optim = Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer=optim, step_size=LR_REDUCE_AFTER, gamma=LR_REDUCTION_FACTOR)
    loss_fn = BCEWithLogitsLoss()
    logger.debug(string=f"Session path: {session_path}")

    if training_configs["checkpoint"]["continue"]:
        checkpoint_type = training_configs["checkpoint"]["type"]
        checkpoint = load_checkpoint(session_path, checkpoint_type, DEVICE, model_name=training_configs["model"]["name"], session_id=session_id, checkpoint_id=training_configs["checkpoint"]["id"])
        model.load_state_dict(checkpoint["model"])
        cp_handler.previous_best_value = checkpoint["score"]
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["sched"])
        START_EPOCH = checkpoint["epoch"]+1


    logger.log(f"Training {MODEL_NAME} starting for {EPOCHS-START_EPOCH} epochs, Learning rate = {LEARNING_RATE}, with Adam optimizer") #! optim must be accessed through configs
    for epoch in range(START_EPOCH, EPOCHS+1):
        model.train()
        logger.log(f"Epoch: {epoch}")
        epoch_cummulative_loss = 0
        steps = 0
        for i, (input_imgs, labels) in enumerate(train_loader):
            # logger.debug(f"input_size: {_256b.shape}")
            optim.zero_grad()

            input_imgs = input_imgs.to(DEVICE)

            output = model(input_imgs)
            pred = torch.sigmoid(output)
            
            loss = loss_fn(output, pred)
            
            loss.backward()
            optim.step()
            
            epoch_cummulative_loss += loss.item()
            steps+=1
        
        scheduler.step()
        avg_train_mse = epoch_cummulative_loss/steps
        #! PERHAPS YOU DON'T NEED TO COMPUTE TRAIN ACCURACY HERE
        test_loss, test_accuracy, train_accuracy = compute_test_metrics(model, loss_fn, DEVICE, test_loader, 1)
        logger.log(f"Average Train Loss = {avg_train_mse:.12f}")
        logger.log(f"Train Accuracy: {train_accuracy}")
        logger.log(f"Test Loss = {test_loss:.12f}")
        logger.log(f"Test Accuracy: {test_accuracy}")

        if cp_handler.check_save_every(epoch):
            logger.checkpoint(f"{SAVE_EVERY} epochs have passed, saving data in last.pth")
            cp_handler.save_model({'model': model.state_dict(),
                                   'optim': optim.state_dict(),
                                   'sched': scheduler.state_dict(),
                                   'test_accuracy': test_accuracy,
                                   'test_loss': test_loss,
                                   'epoch': epoch,
                                   'preds': ["samples"] #probably remove,
                                   }, save_type='last')
            
        if cp_handler.metric_has_improved(test_accuracy):
            logger.checkpoint(f"metric has improved, saving data in best.pth")
            cp_handler.save_model({'model': model.state_dict(),
                                   'optim': optim.state_dict(),
                                   'sched': scheduler.state_dict(),
                                   'test_accuracy': test_accuracy,
                                   'test_loss': test_loss,
                                   'epoch': epoch,
                                   'preds': ["samples"] #probably remove,
                                   }, save_type='best')




def main():
    training_configs, shared_configs = load_configs()
    args = parse_args()
    
    if len(sys.argv) > 1:
        override_configs(training_configs, shared_configs, args)

    dataset_path, output_path = resolve_paths(shared_configs)
    
    session_path, session_id = create_training_environment(output_path)

    logger = Logger(debug_mode=shared_configs["environment"]["debugger_active"], logs_folder_path=os.path.join(session_path, "logs"))



    #TODO: LOG ALL CONFIGS AND SESSION PATH BEFORE STARTING
    log_all_configs(logger=logger, session_path=session_path, training_configs=training_configs, shared_configs=shared_configs)
    # print(os.listdir(session_path))
    train_loader, test_loader = create_data_loaders(dataset_path, training_configs, shared_configs)

    train(train_loader = train_loader,test_loader =  test_loader, training_configs=training_configs, shared_configs=shared_configs, session_path=session_path, session_id=session_id,logger=logger)


if __name__ == "__main__":

    main()