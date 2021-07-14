import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Function
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ACDC_data import onehot2mask


# dir_checkpoint = "D:\PROGRAM\GITHUB\ACDC_UNet\checkpoints"
# dir_results = "D:\PROGRAM\GITHUB\ACDC_UNet\Results"


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                lf = nn.BCEWithLogitsLoss()
                # tot += dice_coeff(mask_pred, true_masks).item()
                tot += lf(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.set_postfix(**{'loss (batch)': tot})
            pbar.update()

    net.train()
    return tot / n_val


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def accuracy(pred, truth):
    return torch.mean(torch.equal(pred, truth))


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def accuracy(pred, gt):
    """(TP + TN) / (TP + FP + FN + TN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp + tn).float() / (tp + fp + tn + fn).float()

    return score.sum() / N


# ===========================================================
def train_net(dataset,
              net,
              device,
              epochs=5,
              batch_size=4,
              lr=0.01,
              val_percent=0.1
              ):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    criterion = nn.BCEWithLogitsLoss()

    # criterion = nn.CrossEntropyLoss()
    epoch_train_loss = []
    epoch_eval_loss = []
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_label = batch['label']

                optimizer.zero_grad()
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_label = true_label.to(device=device, dtype=torch.float32)
                net = net.to(device=device)
                pred = net(imgs)
                loss = criterion(pred, true_label)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])

            plt.figure()
            plt.subplot(1, 3, 1)
            t = pred.cpu()
            t = torch.squeeze(t).detach().numpy()
            t = onehot2mask(t)
            plt.imshow(t)

            plt.subplot(1, 3, 2)
            t = imgs.cpu()
            t = torch.squeeze(t).detach().numpy()
            plt.imshow(t)

            plt.subplot(1, 3, 3)
            t = true_label.cpu()
            t = torch.squeeze(t).detach().numpy()
            t = onehot2mask(t)
            plt.imshow(t)
            plt.show()

            val_score = eval_net(net, val_loader, device)
            epoch_eval_loss.append(val_score)
            epoch_train_loss.append(epoch_loss)

            # if save_cp:
            #     try:
            #         os.mkdir(dir_checkpoint)
            #         logging.info('Created checkpoint directory')
            #     except OSError:
            #         pass
            #     torch.save(net.state_dict(),
            #                dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            #     logging.info(f'Checkpoint {epoch + 1} saved !')

    plt.figure()
    plt.subplot(1, 3, 1)
    t = pred.cpu()
    t = torch.squeeze(t).detach().numpy()
    t = onehot2mask(t)
    plt.imshow(t)

    plt.subplot(1, 3, 2)
    t = imgs.cpu()
    t = torch.squeeze(t).detach().numpy()
    plt.imshow(t)

    plt.subplot(1, 3, 3)
    t = true_label.cpu()
    t = torch.squeeze(t).detach().numpy()
    t = onehot2mask(t)
    plt.imshow(t)
    plt.show()
    # plt.savefig("../Output/Result.png")

    plt.figure()
    plt.plot(np.arange(0, epochs), epoch_train_loss, color='r', label='train_loss')
    plt.plot(np.arange(0, epochs), epoch_eval_loss, color='b', label='evl_loss')
    plt.legend(loc='upper left')
    plt.title('Train and Evaluation Loss')
    plt.show()
    # plt.savefig("../Output/Loss.png")

    torch.save(net.state_dict(), 'best_model.pth')
