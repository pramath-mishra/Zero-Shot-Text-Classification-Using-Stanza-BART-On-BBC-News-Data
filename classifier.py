import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import stanza
import pathlib
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from tabulate import tabulate
from transformers import pipeline


class Classifier:

    def __init__(self, multi_label=False):
        self.multi_label = multi_label
        self.category = ['business', 'entertainment', 'politics', 'sport', 'technology']
        self.nlp = stanza.Pipeline('en', use_gpu=False, processors='tokenize')
        self.model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

    def preprocessing(self, text):
        return " ".join([
            sentence.text
            for sentence in self.nlp(text).sentences
            if sentence.text.find(" ") != -1
        ][:3])

    def predict(self, input_text):
        input_text = self.preprocessing(input_text)
        model_dict = self.model(input_text, self.category, multi_label=self.multi_label)
        result = max(zip(model_dict.get('labels'), model_dict.get('scores')), key=lambda s: s[1])[0]
        score = max(zip(model_dict.get('labels'), model_dict.get('scores')), key=lambda s: s[1])[1]
        return result, score


if __name__ == "__main__":

    # loading bbc datasets
    docs = list()
    labels = list()

    directory = pathlib.Path('bbc')
    label_names = ['business', 'entertainment', 'politics', 'sport', 'technology']

    for label in label_names:
        for file in directory.joinpath(label).iterdir():
            labels.append(label)
            docs.append(file.read_text(encoding='unicode_escape'))
    print(f"bbc news data loaded...\n -sample: {docs[0]}")

    # creating classifier object
    obj = Classifier(multi_label=True)

    # inference
    predictions = [obj.predict(doc) for doc in tqdm(docs)]
    print("inference done...")

    # classification report
    report = metrics.classification_report(y_true=labels, y_pred=list(map(lambda s: s[0], predictions)), labels=label_names, output_dict=True)
    print(f"Accuracy: {round(report['accuracy'], 2)}", file=open("./classification_report.txt", "a"))
    print(f"Macro Avg Precision: {round(report['macro avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))
    print(f"Weighted Avg Precision: {round(report['weighted avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))

    report = pd.DataFrame([
        {
            "label": key,
            "precision": value["precision"],
            "recall": value["recall"],
            "support": value["support"]
        }
        for key, value in report.items()
        if key not in ["accuracy", "macro avg", "weighted avg"]
    ])
    print(
        tabulate(
            report,
            headers="keys",
            tablefmt="psql"
        ),
        file=open("./classification_report.txt", "a")
    )

