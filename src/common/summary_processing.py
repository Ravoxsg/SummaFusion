from nltk.tokenize import sent_tokenize


def pre_rouge_processing(summary, args, highlights=False):
    if highlights:
        summary = "\n".join(sent_tokenize(summary))
    return summary
