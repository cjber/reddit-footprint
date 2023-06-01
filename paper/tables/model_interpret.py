import pandas as pd
from transformers import pipeline


def tbl_interpret():
    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=0,
    )

    nationality_labels = ["British", "English", "Scottish", "Welsh"]
    outs = {"Sentence": [], "Label": [], "Confidence": []}

    for country in ["Scotland", "Wales", "England"]:
        for sentiment in ["hate", "love"]:
            sentence = f"I {sentiment} {country}."
            out = classifier(sentence, nationality_labels)
            outs["Sentence"].append(sentence)
            outs["Label"].append(out["labels"][0])
            outs["Confidence"].append(out["scores"][0])

    return pd.DataFrame(outs).style.format(precision=2).hide(axis="index")


if __name__ == "__main__":
    tbl_interpret().to_latex()
