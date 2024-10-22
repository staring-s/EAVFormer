'''import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# 假设你的数据加载、模型定义等代码已经准备好
from dataprocess.mdataload import MyDataset
from model.avform import fusoinav

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setseeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval(model, test_dataloader, n_classes, device):
    model.eval()
    preds, labelsf = [], []
    all_probabilities = []

    #class_labels = ['surprised', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    #class_labels = ['anger','disgust', 'fear ', 'happy/joy', 'neutral ', 'sad ']
    class_labels = ['positive','negative']

    for video, audio, mel, labels in tqdm(test_dataloader):
        inputs_v = video.to(device)
        inputs_a = audio.to(device)
        inputs_mel = mel.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs_v, inputs_a, inputs_mel)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Assuming the first element of the tuple is the logits

            softmax_outputs = torch.softmax(outputs, dim=1)
            all_probabilities.append(softmax_outputs.cpu().numpy())
            preds.extend(torch.argmax(softmax_outputs, dim=1).cpu().numpy())
            labelsf.extend(labels.cpu().numpy())

    all_probabilities = np.vstack(all_probabilities)
    preds = np.array(preds)
    labelsf = np.array(labelsf)

    precision, recall, f1_score, support = precision_recall_fscore_support(labelsf, preds, average=None, labels=np.arange(n_classes))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labelsf, preds, average='macro')

    true_labels_bin = label_binarize(labelsf, classes=np.arange(n_classes))
    average_precisions = {label: average_precision_score(true_labels_bin[:, i], all_probabilities[:, i]) for i, label in enumerate(class_labels)}
    accuracy = np.mean(preds == labelsf)

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    for i, label in enumerate(class_labels):
        print(f"{label} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1_score[i]:.4f}, AP: {average_precisions[label]:.4f}")

if __name__ == '__main__':
    #setseeds(76)
    #setseeds(219)
    setseeds(183)
    model = fusoinav()
    model.to(device)
    #model_path = 'savemodel/models/ravdess.pth.tar'
    #model_path = r'savemodel/models/cremad.pth.tar'
    model_path = r'savemodel/models/mosei.pth.tar'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    #test_dataloader = DataLoader(MyDataset(mode='test'), batch_size=6, shuffle=False, num_workers=2)
    #cremad
    #test_dataloader = DataLoader(MyDataset(mode='test'), batch_size=10, shuffle=False, num_workers=2)
    #mosei
    test_dataloader = DataLoader(MyDataset(mode='test'), batch_size=6, shuffle=False, num_workers=2)
    #eval(model, test_dataloader, len(['surprised', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']), device)
    eval(model, test_dataloader, len(['positive','negative']), device)
    #eval(model, test_dataloader, len(['anger','disgust', 'fear ', 'happy/joy', 'neutral ', 'sad ']), device)
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from dataprocess.mdataload import MyDataset
from model.avform import fusoinav
from utils.helper_funcs import accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setseeds(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


model = fusoinav()
model.to(device)
test_dataloader = DataLoader(MyDataset(mode='test'), batch_size=10, shuffle=False, num_workers=2)
#model_path = r'savemodel/models/ravdessl1.pth.tar'
#model_path = r'savemodel/models/moseiv1.pth.tar'
model_path = r'savemodel/models/cremadv1.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])


def eval(ii):
    avetest = 0
    model.eval()
    preds, labelsf = [], []
    all_probabilities = []
    #class_labels = ['positive','negative']
    class_labels = ['anger', 'disgust', 'fear ', 'happy/joy', 'neutral ', 'sad ']
    #class_labels = ['surprised', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    n_classes = len(class_labels)

    for video, audio, mel, labels in tqdm(test_dataloader):
        inputsv = video.to(device)
        inputsa = audio.to(device)
        inputmel = mel.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputsv, inputsa, inputmel)
            #outputs = model(inputsv, inputsa, inputmel)
            softmax_outputs = torch.softmax(outputs, dim=1)
            all_probabilities.append(softmax_outputs.cpu().numpy())
            pred = torch.argmax(softmax_outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            labelsf.extend(labels.cpu().numpy())
            acct = accuracy(outputs.detach(), labels.detach(), topk=(1,))[0]
            avetest = avetest + acct.item()


    # Metrics calculation
    avetest = avetest / int(len(test_dataloader))
    precision, recall, f1, _ = precision_recall_fscore_support(labelsf, preds, average='macro')
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(labelsf, preds, average=None)

    all_probabilities = np.vstack(all_probabilities)
    true_labels_bin = label_binarize(labelsf, classes=np.arange(n_classes))
    if true_labels_bin.shape[1] == 1:
        true_labels_bin = np.hstack([1 - true_labels_bin, true_labels_bin])
    average_precisions = {label: average_precision_score(true_labels_bin[:, i], all_probabilities[:, i]) for i, label in
                          enumerate(class_labels)}

    # Output the computed metrics
    print("Overall Accuracy: {:.2f}%".format(avetest))
    print("Macro Precision: {:.4f}, Macro Recall: {:.4f}, Macro F1-Score: {:.4f}".format(precision, recall, f1))
    for label, p, r, f, ap in zip(class_labels, class_precision, class_recall, class_f1, average_precisions.values()):
        print(f"{label} - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, AP: {ap:.4f}")


if __name__ == '__main__':
    #s = 28
    s = 139
    setseeds(s)
    eval(s)

