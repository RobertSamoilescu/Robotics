import torch
import torch.nn.functional as F
import torch.nn as nn
import model
import sys
from logger import Logger
from model import *
from copy import deepcopy
from dataloader import *


# define constants
batch_size = 64
num_workers = 7

# define loaders
train_dataset = UPBDataset("train", augmentation=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

validation_dataset = UPBDataset("test", augmentation=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# initialize logger
logger = Logger('./logs')

# initialize model
net = model.CAE().cuda()
net = net.train()
print(net)

# criterion & optimizer
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4)

best_loss = sys.maxsize


def train(epoch):
    global  best_loss
    running_loss = None

    for i, data in enumerate(train_dataloader, 0):
        # get inputs
        X, Y_gt = data['img'], data['gaussian']

        # send to gpu
        X = X.cuda()
        Y_gt = Y_gt.cuda()

        # zero the parameters gradient
        optimizer.zero_grad()

        # forward, 
        Y = net(X)

        # reshape
        Y = Y.reshape(Y.shape[0], -1)
        Y = F.log_softmax(Y, dim=1)
        Y_gt = Y_gt.reshape(Y_gt.shape[0], -1)

        # compute loss & optimize
        loss = criterion(Y, Y_gt.float())
        loss.backward()
        optimizer.step()

        # update running loss
        running_loss = loss.item() if running_loss is None else 0.9 * running_loss + 0.1 * loss.item()

        # tensor board plots & print
        if i % max(1, (len(train_dataloader) // 50)) == 0:
            # display
            print(' * [%d, %5d] KLDivLoss training: %.6f' % (epoch, i, running_loss))

            # tensorboard plots
            step = epoch * len(train_dataloader) + i
            logger.scalar_summary('KLDivLoss training', loss.item(), step)

            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), step)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

        # test on validation
        if i % max(1, (len(train_dataloader) // 20)) == 0:
            eval_loss = evaluate(step)

            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(net, './checkpoints/best_model')
                print("Model saved")


def evaluate(epoch):
    total_loss = 0 

    net.eval()

    for i, data in enumerate(validation_dataloader, 0):
        # get inputs
        X, Y_gt = data['img'], data['gaussian']

        # send to gpu
        X = X.cuda()
        Y_gt = Y_gt.cuda()

        # forward, backward & optimize
        with torch.no_grad():
            Y = net(X)

            # reshape
            Y = Y.reshape(Y.shape[0], -1)
            Y = F.log_softmax(Y, dim=1)
            Y_gt = Y_gt.reshape(Y_gt.shape[0], -1)

            loss = criterion(Y, Y_gt.float())
        
        total_loss += loss.item() * X.shape[0]

    print("\t* [%d] KLDivLoss validation: %.6f" % (epoch, total_loss / len(validation_dataset)))
    logger.scalar_summary("KLDivLoss validation", total_loss / len(validation_dataset), epoch)

    net.train()
    return total_loss / len(validation_dataset)


def main():
    for epoch in range(1, 100000):
        train(epoch)

if __name__ == "__main__":
    main()

