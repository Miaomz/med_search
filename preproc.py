# Read and preprocess the XML data
import xml.etree.ElementTree as ET
from random import random, shuffle
import json
from datasets import Dataset
from sentence_transformers import InputExample


def filter_title(raw_text):
    if raw_text is None or raw_text == '':
        return raw_text

    if len(raw_text) > 2 and raw_text[0] == '[' and raw_text[-1] == ']':
        return raw_text[1:-1]
    return raw_text


def load_dict():
    ret = {'title': [], 'abstract': []}

    tree = ET.parse('pubmed21n0001.xml')
    root = tree.getroot()
    articles = root.findall('./PubmedArticle/MedlineCitation/Article')
    for article in articles:
        title = article.find('./ArticleTitle')
        abstract = article.find('./Abstract/AbstractText')

        if abstract is not None:
            ret['title'].append(filter_title(title.text))
            ret['abstract'].append(abstract.text)
    return ret


def load_dataset():
    loaded_dict = load_dict()
    loaded_len = len(loaded_dict['title'])

    # Generate half positive, half negative data items
    dataset_dict = {'texts': [], 'label': []}
    for i in range(0, loaded_len):
        dataset_dict['texts'].append([loaded_dict['title'][i], loaded_dict['abstract'][i]])
        dataset_dict['label'].append(0.9)
    for i in range(0, loaded_len):
        offset = int(random() * (loaded_len - 2)) + 1
        wrong_abstract = (offset + i) % loaded_len
        dataset_dict['texts'].append([loaded_dict['title'][i], loaded_dict['abstract'][wrong_abstract]])
        dataset_dict['label'].append(0.1)

    ret = Dataset.from_dict(dataset_dict)
    return ret


def load_input_examples(is_shuffle=True, cached=False, load_path='./data/input_examples.json'):
    if cached and load_path is not None:
        in_file = open(load_path, 'r')
        # The format of data is '[[title, abstract, label]]'
        json_data = json.load(in_file)
        in_file.close()

        ret = []
        for line in json_data:
            ret.append(InputExample(texts=line[:2], label=line[2]))
        return ret

    loaded_dict = load_dict()
    loaded_len = len(loaded_dict['title'])

    # Generate half positive, half negative data items
    ret = []
    for i in range(0, loaded_len):
        ret.append(InputExample(texts=[loaded_dict['title'][i], loaded_dict['abstract'][i]], label=0.9))

    for i in range(0, loaded_len):
        offset = int(random() * (loaded_len - 2)) + 1
        wrong_abstract = (offset + i) % loaded_len
        ret.append(InputExample(texts=[loaded_dict['title'][i], loaded_dict['abstract'][wrong_abstract]], label=0.1))

    if is_shuffle:
        shuffle(ret)

    if not cached and load_path is not None:
        out_file = open(load_path, 'w')
        json_data = []
        for example in ret:
            json_data.append([example.texts[0], example.texts[1], example.label])
        json.dump(json_data, out_file)
        out_file.close()
    return ret


if __name__ == '__main__':
    my_dataset = load_dataset()
    print(my_dataset['texts'][0])
    print(my_dataset.train_test_split(test_size=0.1))
    load_input_examples(cached=False)

