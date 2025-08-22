import os

_pre_cwd = os.path.realpath(os.getcwd())

import argparse
import json
import pickle

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml

torch.multiprocessing.set_sharing_strategy("file_system")

# local modules
import robust_scene_change_detect.datasets as datasets
import robust_scene_change_detect.models as models
import robust_scene_change_detect.torch_utils as torch_utils
from py_utils import utils, utils_torch

# setting environment
if not utils.is_connect_to_network():
    os.environ["WANDB_MODE"] = "offline"

# global variable
_dry = False
_seed = 123
_device = "cuda"
_verbose = False
_prefix = " " * 4


def xprint(*args):
    if _verbose:
        print(_prefix, *args)


def train_one_epoch(
    model,
    criterion,
    scaler,
    optimizer,
    data_loader,
    verbose=False,
):

    def xxprint(*x):
        if verbose:
            print(_prefix, *x)

    xxprint("")
    xxprint("*** train one epoch ***")

    # make any layer that requires_grad to training mode
    # and others in evaluation mode
    utils_torch.set_grad_required_layer_train(model)

    prog = utils.ProgressTimer(verbose=False)
    prog.tic(len(data_loader))

    total_loss = 0
    data_inds = []
    batch_losses = []

    for idx, (inds, (input_1, input_2, targets)) in enumerate(data_loader):

        data_inds.append(inds)

        if idx == 0:
            xxprint("input_1: ", input_1.shape)
            xxprint("input_2: ", input_2.shape)
            xxprint("targets: ", targets.shape)

        input_1 = input_1.to(_device)
        input_2 = input_2.to(_device)
        targets = targets.to(_device)

        optimizer.zero_grad()

        outputs = model(input_1, input_2)  # (n, 504, 504, 2)
        transform_outputs = outputs.permute(0, 3, 1, 2)

        if idx == 0:
            xxprint("output: ", outputs.shape, "->", transform_outputs.shape)

        # Resize targets to match model output size
        target_size = transform_outputs.shape[2:]  # Get spatial dimensions (504, 504)
        if targets.shape[1:] != target_size:
            targets = F.interpolate(
                targets.unsqueeze(1).float(), 
                size=target_size, 
                mode='nearest'
            ).squeeze(1)
            if idx == 0:
                xxprint("targets resized to:", targets.shape)

        # not sure why it need to specify long format
        targets = targets.to(torch.long)

        loss = criterion(transform_outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())

        msg = f"Train Epoch: [{idx}/{len(data_loader)}]"
        msg += f"\tLoss: {loss.item():.6f}"

        if idx % 20 == 0 or _dry:  # log_interval can be set as needed
            xprint(msg)

        prog.toc()

        if _dry:
            break

    xxprint("*** train one epoch ***")
    xxprint("")

    # Calculate average loss over the epoch
    average_loss = total_loss / len(data_loader)

    data_ind = torch.concat(data_inds)
    return average_loss, data_ind, batch_losses, prog.total_seconds


def evaluate_with_resize(model, dataset, verbose=False, prefix="", return_duration=True, dry_run=False, device="cuda"):
    """Wrapper for evaluation that handles target resizing to match model output."""
    import time
    start_time = time.time()
    
    # Get model output shape by doing a forward pass with dummy data
    model.eval()
    with torch.no_grad():
        # Get a sample from the dataset to determine input/output shapes
        sample_data = next(iter(dataset))
        
        # Debug: print the actual structure
        if verbose:
            print(f"Debug: sample_data has {len(sample_data)} elements")
            print(f"Debug: sample_data types: {[type(x) for x in sample_data]}")
        
        # Handle different dataset formats more carefully
        if len(sample_data) == 2:
            # Could be ((input1, input2), target) or (input1, input2)
            first_elem, second_elem = sample_data
            if isinstance(first_elem, tuple) and len(first_elem) == 2:
                # Format: ((input1, input2), target)
                (sample_input1, sample_input2), sample_target = sample_data
            else:
                # Format: (input1, input2) - no target
                sample_input1, sample_input2 = sample_data
                sample_target = None
        elif len(sample_data) == 3:
            # Could be (ind, input1, input2) or (input1, input2, target)
            first_elem = sample_data[0]
            if torch.is_tensor(first_elem) and first_elem.dim() == 1:
                # Likely indices, format: (ind, input1, input2)
                sample_ind, sample_input1, sample_input2 = sample_data
                sample_target = None
            else:
                # Format: (input1, input2, target)
                sample_input1, sample_input2, sample_target = sample_data
        elif len(sample_data) == 4:
            # Format: (ind, input1, input2, target)
            sample_ind, sample_input1, sample_input2, sample_target = sample_data
        else:
            raise ValueError(f"Unexpected dataset format with {len(sample_data)} elements")
        
        sample_input1 = sample_input1[:1].to(device)  # Take only first sample
        sample_input2 = sample_input2[:1].to(device)
        
        sample_output = model(sample_input1, sample_input2)
        output_shape = sample_output.shape[1:3]  # Get spatial dimensions
    
    # Create a wrapper dataset that resizes targets
    class ResizedTargetDataset:
        def __init__(self, original_dataset, target_size):
            self.original_dataset = original_dataset
            self.target_size = target_size
            # Add dataset attribute for compatibility with evaluation function
            self.dataset = original_dataset
        
        def __len__(self):
            return len(self.original_dataset)
        
        def __iter__(self):
            for item in self.original_dataset:
                if len(item) == 2:
                    first_elem, second_elem = item
                    if isinstance(first_elem, tuple) and len(first_elem) == 2:
                        # Format: ((input1, input2), target)
                        (input1, input2), target = item
                        if target.shape[1:] != self.target_size:
                            target = F.interpolate(
                                target.unsqueeze(1).float(),
                                size=self.target_size,
                                mode='nearest'
                            ).squeeze(1)
                        yield (input1, input2), target
                    else:
                        # Format: (input1, input2) - no target, pass through
                        yield item
                elif len(item) == 3:
                    first_elem = item[0]
                    if torch.is_tensor(first_elem) and first_elem.dim() == 1:
                        # Format: (ind, input1, input2)
                        yield item
                    else:
                        # Format: (input1, input2, target)
                        input1, input2, target = item
                        if target.shape[1:] != self.target_size:
                            target = F.interpolate(
                                target.unsqueeze(1).float(),
                                size=self.target_size,
                                mode='nearest'
                            ).squeeze(1)
                        yield input1, input2, target
                elif len(item) == 4:
                    # Format: (ind, input1, input2, target)
                    ind, input1, input2, target = item
                    if target.shape[1:] != self.target_size:
                        target = F.interpolate(
                            target.unsqueeze(1).float(),
                            size=self.target_size,
                            mode='nearest'
                        ).squeeze(1)
                    yield ind, input1, input2, target
                else:
                    # Unknown format, pass through
                    yield item
    
    resized_dataset = ResizedTargetDataset(dataset, output_shape)
    
    # Call the original evaluation function
    stats = torch_utils.image_change_detection_evaluation(
        model, resized_dataset, verbose=verbose, prefix=prefix, 
        return_duration=False, dry_run=dry_run, device=device
    )
    
    if return_duration:  
        duration = time.time() - start_time
        return stats, duration
    return stats


def main(args):

    if _verbose:
        print(json.dumps(args, indent=4))

    utils_torch.seed_everything(_seed)

    ######### load checkpoints #########
    checkpoints = args["model"]["checkpoints"]
    if not os.path.exists(checkpoints) or not os.path.isfile(checkpoints):
        raise ValueError(f"checkpoints not found: {checkpoints}")

    checkpoints = torch.load(checkpoints, map_location=_device)

    xprint("")
    xprint("use the checkpoints from pretrained model: ")
    xprint("")

    if _verbose:
        print(json.dumps(checkpoints["args"], indent=4))

    pretrain_path = os.path.join(args["wandb"]["output-path"], "pretrain.pth")
    torch.save(checkpoints, pretrain_path)

    # update the arguments through pretraining model
    for i, j in checkpoints["args"]["model"].items():
        if i in args["model"]:
            continue
        args["model"][i] = j

    model = models.get_model(**args["model"]).to(_device)
    model = nn.DataParallel(model)

    model = utils_torch.load_grad_required_state(model, checkpoints["model"])
    ####################################

    utils_torch.seed_everything(_seed)

    ######### get batchify datasets #########
    opt = args["dataset"]

    if opt["dataset"] == "PSCD":
        trainset = datasets.get_PSCD_training_datasets(**opt)

        pscd_opts = {
            "use_mask_t0": True,
            "use_mask_t1": False,
        }

        valset_1 = datasets.get_dataset("PSCD", mode="val", **pscd_opts)
        testset_1 = datasets.get_dataset("PSCD", mode="test", **pscd_opts)

        wrapper = datasets.wrap_eval_dataset(opt, shuffle=False)
        valset_1 = wrapper(valset_1)
        testset_1 = wrapper(testset_1)

    elif opt["dataset"] == "CUSTOM":
        # Handle your custom dataset
        trainset = datasets.get_CUSTOM_training_datasets(**opt)

        # For validation and test, you can either:
        # Option 1: Use your test set for both validation and testing
        testset_raw = datasets.get_dataset("CUSTOM", mode="test")
        
        # Split test set into validation and test (optional)
        # You can modify this split ratio as needed
        test_size = len(testset_raw)
        val_size = test_size // 2  # Use half for validation, half for testing
        
        indices = list(range(test_size))
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]
        
        valset_raw = testset_raw.loc(val_indices)
        testset_raw = testset_raw.loc(test_indices)
        
        wrapper = datasets.wrap_eval_dataset(opt, shuffle=False)
        valset_1 = wrapper(valset_raw)
        testset_1 = wrapper(testset_raw)

    else:
        raise NotImplementedError(f"Dataset {opt['dataset']} not implemented")
    #########################################

    utils_torch.seed_everything(_seed)

    ######### optimizer #########
    opt = args["optimizer"]
    epochs = opt["epochs"]
    warmup_epoch = opt["warmup-epoch"]
    learn_rate = opt["learn-rate"]
    loss_weight = opt["loss-weight"]
    lr_scheduler = opt["lr-scheduler"]
    grad_scaler = opt["grad-scaler"]

    # set up loss weight
    weight = None
    if loss_weight:
        # this value comes from examples/datasets.ipynb
        weight = torch.tensor([0.025, 0.975]).float().to(_device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    scaler = None
    if grad_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    if lr_scheduler.lower() == "none":
        lr_scheduler = None

    lr_scheduler = utils_torch.CustomizedLRScheduler(
        optimizer,
        start_scale=0.0,
        warmup_epoch=warmup_epoch,
        final_scale=0.2 * learn_rate,
        total_epoch=epochs,
        mode=lr_scheduler,
    )
    #############################

    ######### training and evaluation #########
    opt = args["evaluation"]
    freq_train = opt["trainset"]

    best_val_f1_score = -1

    for epoch in range(epochs):

        verbose = (epoch == 0) and _verbose
        utctime = utils.get_utc_time()

        xprint(f"===== running {epoch} epoch =====")
        xprint(utctime)

        # traiing
        loss, data_inds, batch_losses, train_time = train_one_epoch(
            model, criterion, scaler, optimizer, trainset, verbose
        )
        lr_scheduler.step()

        # logging batch loss
        N = len(batch_losses)
        for n, batch_loss in enumerate(batch_losses):
            wandb.log(
                {
                    "batch_report": {
                        "epoch": epoch,
                        "batch": epoch * N + n,
                        "loss": batch_loss,
                    },
                }
            )

        logs = {
            "loss": loss,
            "epoch": epoch,
            "time.train": train_time,
        }

        checkpoint = {
            "model": utils_torch.get_grad_required_state(model),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "data_inds": data_inds,
        }

        # evaluation on trainset
        if freq_train > 0 and (epoch + 1) % freq_train == 0:
            # Create evaluation version of trainset without indices wrapper
            dataset_opt = args["dataset"]  # Use the correct dataset options
            if dataset_opt["dataset"] == "PSCD":
                trainset_eval = datasets.get_dataset("PSCD", mode="train", 
                                                   use_mask_t0=True, use_mask_t1=False)
                wrapper = datasets.wrap_eval_dataset(dataset_opt, shuffle=False)
                trainset_eval = wrapper(trainset_eval)
            elif dataset_opt["dataset"] == "CUSTOM":
                trainset_eval = datasets.get_dataset("CUSTOM", mode="train")
                wrapper = datasets.wrap_eval_dataset(dataset_opt, shuffle=False)
                trainset_eval = wrapper(trainset_eval)
            else:
                # Skip trainset evaluation for unsupported datasets
                trainset_eval = None
            
            if trainset_eval is not None:
                statics, time = evaluate_with_resize(
                    model,
                    trainset_eval,
                    verbose=_verbose,
                    prefix="Trainset: ",
                    return_duration=True,
                    dry_run=_dry,
                    device=_device,
                )
                logs["evaluation.train"] = statics
                logs["time.eval.train"] = time

        # evaluation on testset
        statics, time = evaluate_with_resize(
            model,
            testset_1,
            verbose=_verbose,
            prefix="Testset: ",
            return_duration=True,
            dry_run=_dry,
            device=_device,
        )

        logs["evaluation.test"] = statics
        logs["time.eval.test"] = time

        # evaluation on valset
        statics, time = evaluate_with_resize(
            model,
            valset_1,
            verbose=_verbose,
            prefix="Valset : ",
            return_duration=True,
            dry_run=_dry,
            device=_device,
        )

        logs["evaluation.val"] = statics
        logs["time.eval.val"] = time

        path = str(epoch) + ".layer"

        log_path = os.path.join(
            args["wandb"]["output-path"], "logs", path + ".pkl"
        )

        checkpoint_path = os.path.join(
            args["wandb"]["output-path"], "checkpoints", path + ".pth"
        )

        wandb.log(logs)
        with open(log_path, "wb") as fd:
            pickle.dump(logs, fd)

        checkpoint["logs"] = logs

        save_freq = args["wandb"]["save-checkpoint-freq"]
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(checkpoint, checkpoint_path)

        # save the best validation - handle both dict and tuple returns
        val_f1_score = 0.0
        if isinstance(logs["evaluation.val"], dict):
            val_f1_score = logs["evaluation.val"]["f1_score"]
        elif isinstance(logs["evaluation.val"], tuple) and len(logs["evaluation.val"]) >= 3:
            # Assuming f1_score is the 3rd element in tuple (precision, recall, f1_score, ...)
            val_f1_score = logs["evaluation.val"][2]
        elif hasattr(logs["evaluation.val"], 'f1_score'):
            val_f1_score = logs["evaluation.val"].f1_score
        
        if val_f1_score > best_val_f1_score:
            best_val_f1_score = val_f1_score
            best_val_path = os.path.join(
                args["wandb"]["output-path"], "best.val.pth"
            )
            torch.save(checkpoint, best_val_path)

        if _dry:
            break

    # save the last checkpoint
    last_checkpoint_path = os.path.join(
        args["wandb"]["output-path"], "last.pth"
    )
    torch.save(checkpoint, last_checkpoint_path)
    ###########################################


def get_output_path_by_utc(prefix="", suffix=""):

    # specify directory by using utc time
    # YYYY-MM-DD.hh-mm-ss
    utctime = utils.get_utc_time()
    output_folder = prefix + utctime + suffix
    output_path = os.path.join(os.getcwd(), "output", output_folder)

    # try to avoid race-condition(save checkpoint in the same output_path)
    cnt = 0
    while os.path.exists(output_path):
        if cnt == 0:
            output_path += "." + str(cnt)
            continue
        cnt += 1
        output_path = ".".join((output_path.split(".")[:-1] + [str(cnt)]))

    return output_path


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML configuration file")
    args = parser.parse_args()

    config = args.config
    config = (
        config
        if config == os.path.abspath(config)
        else os.path.join(_pre_cwd, args.config)
    )

    with open(config, "r") as fd:
        args = yaml.safe_load(fd)

    return args


if __name__ == "__main__":

    args = parse_args()

    # setting global variable
    env = args["environment"]
    _dry = env["dry"]
    _seed = env["seed"]
    _verbose = env["verbose"]
    _device = env["device"]

    # setting output directory
    output_path = args["wandb"]["output-path"]
    if len(output_path) == 0:
        output_path = get_output_path_by_utc()
    args["wandb"]["output-path"] = output_path

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)

    logs = {
        "project": args["wandb"]["project"],
        "name": args["wandb"]["name"],
        "config": args,
        "dir": output_path,
    }

    # save input arguments for future reference
    with open(os.path.join(output_path, "args.json"), "w") as fd:
        json.dump(args, fd, indent=4)

    if _dry:
        os.environ["WANDB_MODE"] = "offline"

    wandb.login()
    wandb.init(**logs)

    main(args)
    wandb.init(**logs)

    main(args)
