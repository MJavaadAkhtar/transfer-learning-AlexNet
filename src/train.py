from model import *

def get_accuracy(model, data):
    '''
    @model: The model we are running training accuracy on.
    @data: A batch of images to check the accuracy
    @return: The probability of accuracy
    '''

    loader = torch.utils.data.DataLoader(data, batch_size=32)
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total



def train(model, train_data, batch_size=32, weight_decay=0.0,
          learning_rate=0.001, num_epochs=100, checkpoint_path=None):
    '''
    @model: The MLP model we are training
    @batch_size: The batch size to use
    @weight_decay: The weight decay parameter for Adam optimizer
    @learning rate: The learning rate for the adam optimizer
    @nnum_epochs: The number of epochs to run
    '''

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            if imgs.size()[0] < batch_size:
                continue

            model.train()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n += 1

        # save the current training information
        loss = float(loss)/batch_size
        tacc = get_accuracy(model, train_data)
        # vacc = get_accuracy(model, valid_data)
        # print("Iter %d; Loss %f; Train Acc %.3f; Val Acc %.3f" % (n, loss, tacc, vacc))
        print("Iter %d; Loss %f; Train Acc %.3f;" % (n, loss, tacc))

        iters.append(n)
        losses.append(loss)
        train_acc.append(tacc)

        if (checkpoint_path is not None) and epoch % 20 == 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch))
        # val_acc.append(vacc)

    # plotting
    plt.title("Learning Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Learning Curve")
    plt.plot(iters, train_acc, label="Train")

    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))

mlp = MLP()
train(mlp, train_data_features, batch_size=32)