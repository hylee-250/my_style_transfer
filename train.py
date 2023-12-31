import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import StyleSpeechLossMain
from dataset import Dataset

from evaluate import evaluate

import pdb
import wandb

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my_style_transfer"
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backward(model, optimizer, total_loss, step, grad_acc_step, grad_clip_thresh):
    total_loss = total_loss / grad_acc_step
    total_loss.backward()
    if step % grad_acc_step == 0:
        # Clipping gradients to avoid gradient explosion
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

        # Update weights
        optimizer.step_and_update_lr()
        optimizer.zero_grad()


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train_filtered.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer_main, _ = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = StyleSpeechLossMain(preprocess_config, model_config, train_config).to(device)
    print("Number of StyleSpeech Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                # output = (None, None, *model(*(batch[2:-5])))
                output = model(*(batch[2:-5]))

                '''
                output
                    mel_out,p_predictions,e_predictions,log_d_predictions,
                    d_rounded,src_masks,mel_masks,src_lens,mel_lens,
                    z_pf,ldj_pf, postflow,
                
                batch 
                    ids,raw_texts,speakers,texts,text_lens,
                    max(text_lens),mels,mel_lens,max(mel_lens),
                    pitches,energies,durations,raw_quary_texts,quary_texts,quary_text_lens,
                    max(quary_text_lens),quary_durations,
                '''

                # Cal Loss            
                losses_1 = Loss(batch, output)
                total_loss = losses_1[0]

                # Backward
                backward(model, optimizer_main, total_loss, step, grad_acc_step, grad_clip_thresh)

                if step % log_step == 0:
                    losses = [l.item() for l in (losses_1+tuple([torch.zeros(1).to(device) for _ in range(3)]))]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Post flow:{:.4f} ".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output[2:],
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder, len(losses))
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer_main": optimizer_main._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default='/home/work/hylee/my_style_transfer/config/LibriTTS/preprocess.yaml',
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default='/home/work/hylee/my_style_transfer/config/LibriTTS/model.yaml', help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default='/home/work/hylee/my_style_transfer/config/LibriTTS/train.yaml', help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
