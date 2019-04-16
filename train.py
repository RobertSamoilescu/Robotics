import torch
import model
from logger import Logger
from model import *


if __name__ == "__main__":
    # read all data
    dataset = UPBDataset("dataset")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)

    # initialize logger
    logger = Logger('./logs')

    # initialize model
    net = model.VGGBased(2).cuda()
    net = net.train()

    # criterion & optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # start training
    running_loss = None
    best_net = None
    best_accuracy = -1

    for epoch in range(1, 100000):
        for i, data in enumerate(dataloader, 0):
            # get inputs
            X, Y_gt = data['img'], data['steer_coord']

            # send to gpu
            X = X.cuda()
            Y_gt = Y_gt.cuda()

            # zero the parameters gradient
            optimizer.zero_grad()

            # forward, backward & optimize
            Y = net(X)
            loss = criterion(Y, Y_gt)
            loss.backward()
            optimizer.step()

            # update running loss
            running_loss = loss.item() if running_loss is None else 0.9 * running_loss + 0.1 * loss.item()

            # tensor board plots & print
            if i % max(1, (len(dataloader) // 8)) == 0:
                # display
                print('[%d, %5d] loss: %.6f' % (epoch, i, running_loss))

                # tensorboard plots
                step = epoch * len(dataloader) + i
                logger.scalar_summary('Loss', loss.item(), step)

                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

        # save best model every 5 epochs
        if epoch % 10 == 0:
            torch.save(best_net, './checkpoints/vgg2d_' + str(epoch))
            print("Model saved")