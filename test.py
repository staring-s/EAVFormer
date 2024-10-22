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



model = fusoinav()
model.to(device)
test_dataloader = DataLoader(MyDataset(mode='test'), batch_size=10, shuffle=False, num_workers=2)
model_path = r'savemodel/models/ravdess.pth.tar'
#model_path = r'savemodel/models/mosei.pth.tar'
#model_path = r'savemodel/models/cremad.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])


def eval():
    avetest = 0
    model.eval()
    preds, labelsf = [], []
    all_probabilities = []
    #class_labels = ['positive','negative']
    #class_labels = ['anger', 'disgust', 'fear ', 'happy/joy', 'neutral ', 'sad ']
    class_labels = ['surprised', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
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
    eval()

