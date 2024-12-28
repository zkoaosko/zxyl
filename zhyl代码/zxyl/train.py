import copy
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def data_transforms(phase):
    if phase == TRAIN:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if phase == VAL:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if phase == TEST:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return transform



def train_test_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch + 1, num_epochs))
        print("=" * 10)

        for phase in [TRAIN, TEST]:
            if phase == TRAIN:
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for data in tqdm(dataloaders[phase]):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if  phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                test_losses.append(epoch_loss)
                test_accuracies.append(epoch_acc)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return train_losses, train_accuracies, test_losses, test_accuracies, model


def eval_model():

    running_correct = 0.0
    running_total = 0.0
    true_labels = []
    pred_labels = []

    phase = TEST

    with torch.no_grad():
        for data in dataloaders[phase]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels.extend(labels.cpu().numpy())
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            pred_labels.extend(preds.cpu().numpy())

            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()

        acc = running_correct / running_total

        precision = precision_score(true_labels, pred_labels, average='macro', zero_division=1)
        recall = recall_score(true_labels, pred_labels, average='macro', zero_division=1)
        f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=1)

        print(f'{phase} Precision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1:.4f}')

    return (true_labels, pred_labels, running_correct, running_total, acc)



def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    """ 绘制训练和验证的损失和准确率曲线 """
    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='TEST Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='TEST Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    EPOCHS = 12
    data_dir = "chest_xray"
    TEST = 'test'
    TRAIN = 'train'
    VAL = 'val'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms(x))
                      for x in [TRAIN, VAL, TEST]}

    dataloaders = {TRAIN: torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size=64, shuffle=True),
                   VAL: torch.utils.data.DataLoader(image_datasets[VAL], batch_size=1, shuffle=True),
                   TEST: torch.utils.data.DataLoader(image_datasets[TEST], batch_size=1, shuffle=True)}

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, TEST]}
    classes = image_datasets[TRAIN].classes
    class_names = image_datasets[TRAIN].classes

    model_pre = models.vgg16()
    model_pre.load_state_dict(torch.load("vgg16-397923af.pth"))

    for param in model_pre.features.parameters():
        param.required_grad = False

    num_features = model_pre.classifier[6].in_features
    features = list(model_pre.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, len(class_names))])
    model_pre.classifier = nn.Sequential(*features)

    model_pre = model_pre.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_pre.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.01)
    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, train_accuracies, test_losses, test_accuracies, model = train_test_model(model_pre, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)

    true_labels, pred_labels, running_correct, running_total, acc = eval_model()

    print("Total Correct: {}, Total Test Images: {}".format(running_correct, running_total))
    print("Test Accuracy: ", acc)

    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    ax = sns.heatmap(cm, annot=True, fmt="d")


    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies )