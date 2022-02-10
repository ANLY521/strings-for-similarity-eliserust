# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """
    texts = []
    labels = []

    with open(data_file, 'r') as d:
        for line in d:
            if line.strip():
                texts.append(line.lower().strip())
                labels.append(line.lower().strip())
    return texts, labels


    return texts, labels