import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import scipy.io as sio
import time

################################################################################################################
# Hyper Parameters
input_size = 3*32*32
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 500
learning_rate = 0.001

################################################################################################################
train_dataset = dsets.SVHN(root='./data_svhn', split='train', download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

test_dataset = dsets.SVHN(root='./data_svhn', split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ['0','1','2','3','4','5','6','7','8','9']

################################################################################################################
# Plot of data
mat=sio.loadmat('data_svhn/train_32x32.mat')
data=mat['X']
label=mat['y']
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.title(label[i][0])
    plt.imshow(data[...,i])
    plt.axis('off')
plt.show()

# Plot of batch
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
plt.show()

print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

################################################################################################################


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


################################################################################################################
net = Net(input_size, hidden_size, num_classes)
net.cuda()

################################################################################################################
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
################################################################################################################

loss_list = []

start = time.time()

# Train the Model
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

end = time.time()
total_time = end - start

################################################################################################################
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 3* 32 * 32)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()

print('Accuracy of the network on the 26032 test images: %d %%' % (100 * correct / total))

################################################################################################################

_, predicted = torch.max(outputs.data, 1)

print('Predicted label: ', ' '.join('%5s' % predicted[j].cpu().numpy() for j in range(20)))
print('Actual label: ', ' '.join('%5s' % labels[j].cpu().numpy() for j in range(20)))

################################################################################################################

# There is bug here find it and fix it
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

test_labels_list = []
test_predicted_list = []

all_test_labels = []
all_test_predicted = []

for images, labels in test_loader:
    images = Variable(images.view(-1, 3 * 32 * 32)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)

    for i in range(len(labels)):
        test_labels_list.append(classes[labels[i]])
        test_predicted_list.append(classes[predicted.cpu()[i]])
        all_test_labels.append(labels[i])
        all_test_predicted.append(predicted.cpu()[i])

    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.metrics import classification_report

print('Classification report:\n', classification_report(test_labels_list, test_predicted_list))

print("Precision: %1.3f" % precision_score(test_labels_list, test_predicted_list, average='macro'))
print("Recall: %1.3f" % recall_score(test_labels_list, test_predicted_list, average='macro'))
print("F1: %1.3f\n" % f1_score(test_labels_list, test_predicted_list, average='macro'))

print('Training time:', total_time)

# Save the Model
torch.save(net.state_dict(), 'model.pkl')

################################################################################################################

# Plot loss
plt.plot(loss_list)
plt.grid(True, which='both')
plt.ylabel('loss')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()

################################################################################################################

# Plot ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
y_label = label_binarize(all_test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_predict = label_binarize(all_test_predicted, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

################################################################################################################

for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

################################################################################################################

def confusion_matrix_plot(y_test, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


confusion_matrix_plot(test_labels_list, test_predicted_list, classes, normalize=False)

confusion_matrix_plot(test_labels_list, test_predicted_list, classes, normalize=True)

