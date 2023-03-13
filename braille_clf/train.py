import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

from config.config import BrailleParams
from data.dataio import Braille_Dataset

# load data and model configs


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/config_tacto.yaml")
    args = parser.parse_args()
    return args


def load_config(cfg_file):
    cfg = BrailleParams()
    cfg.load(cfg_file)
    return cfg


# cfg = BrailleParams()
# cfg_file = "config/config_tacto.yaml"
# cfg.load(cfg_file)


def load_data(cfg):
    train_dataset = Braille_Dataset(cfg, mode="train")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    test_dataset = Braille_Dataset(cfg, mode="test")
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    class_names = train_dataset.classes
    return train_dataset, train_dataloader, test_dataset, test_dataloader, class_names


def load_model(cfg, num_outputs: int):
    if cfg.model == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        raise NotImplementedError("Model not implemented")

    for i, child in enumerate(model.children()):
        if not cfg.fine_tune:
            if i < 4:
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_features // 2),
        nn.ReLU(),
        nn.Linear(num_features // 2, num_outputs),
        nn.Sigmoid(),
    )
    if cfg.checkpoint != "":
        checkpoint_path = os.path.join("./checkpoints/", cfg.checkpoint)
        model.load_state_dict(torch.load(checkpoint_path))
    return model


def create_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def test_model(model, test_dataloader, device):
    y_pred, y_true = [], []
    """ Testing Phase """
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        for inputs, labels, _ in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            y_pred.extend(preds.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())
    return y_pred, y_true, running_corrects


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (
        train_dataset,
        train_dataloader,
        test_dataset,
        test_dataloader,
        class_names,
    ) = load_data(cfg)
    model = load_model(cfg, len(class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    checkpoint_dir = "./checkpoints/"
    log_dir = "./tb_logs/{0}/".format(cfg.name_source_data)
    create_dir(log_dir)
    create_dir(checkpoint_dir)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(cfg.n_epochs):  # (loop for every epoch)
        # print("Epoch {} running".format(epoch)) #(printing message)
        """Training Phase"""
        model.train()
        running_loss = 0.0
        running_corrects = 0
        # load a batch data of images
        for i, (inputs, labels, _) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # get loss value and update the network weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100.0
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        if epoch % 10 == 0:
            print(
                "[Train {} #{}] Loss: {:.4f} Acc: {:.4f}%".format(
                    cfg.name_source_data, epoch, epoch_loss, epoch_acc
                )
            )

        if epoch % 10 == 0:
            _, _, running_corrects = test_model(model, test_dataloader, device)
            epoch_acc = running_corrects / len(test_dataset) * 100.0
            writer.add_scalar("Accuracy/Real", epoch_acc, epoch)
            print("[Test (real) #{}] Acc: {:.4f}%".format(epoch, epoch_acc))

    # save model
    if not cfg.fine_tune:
        filename = "{0}/clf_{1}.pth".format(checkpoint_dir, cfg.name_source_data)
    else:
        filename = "{0}/clf_{1}_{2}.pth".format(
            checkpoint_dir, cfg.name_source_data, cfg.p_data_fine_tune
        )
    torch.save(model.state_dict(), filename)

    # final test on real dataset
    y_pred, y_true, running_corrects = test_model(model, test_dataloader, device)
    acc = running_corrects / len(test_dataset) * 100.0
    print(f"Final Test accuracy = {acc:.4f}%")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=test_dataset.classes,
            target_names=test_dataset.letters,
        )
    )


if __name__ == "__main__":
    args = load_args()
    cfg = load_config(args.cfg)
    main(cfg)
