from datetime import datetime
import os
from scipy import spatial
import json
import numpy as np
from preproc import load_dict, load_input_examples
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader


def test():
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    input_examples = load_input_examples()
    print(input_examples[0].texts[0])
    print(input_examples[0].texts[1])
    print(input_examples[1].texts[1])

    embedding0 = model.encode(input_examples[0].texts[0])
    embedding1 = model.encode(input_examples[0].texts[1])
    embedding2 = model.encode(input_examples[1].texts[0])
    embedding3 = model.encode(input_examples[1].texts[1])

    print(1-spatial.distance.cosine(embedding0, embedding1))
    print(1-spatial.distance.cosine(embedding0, embedding3))
    print(1-spatial.distance.cosine(embedding2, embedding1))
    print(1-spatial.distance.cosine(embedding2, embedding3))


def main():
    # Define the model by loading a pre-trained model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    # fixed partition of the train, dev, and test data sets
    input_examples = load_input_examples(cached=True)
    train_examples = input_examples[:20000]
    dev_examples = input_examples[20000:25000]
    test_examples = input_examples[25000:]

    # save the test results of pretrained model first
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pre_save_path = './output/pretrained-' + current_time
    model_save_path = './output/finetuned' + current_time
    if not os.path.exists(pre_save_path):
        os.mkdir(pre_save_path)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='dev')
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='test')
    test_evaluator(model, output_path=pre_save_path)

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500, output_path=model_save_path)
    test_evaluator(model, output_path=model_save_path)


def predict(model_path, queries, cache=True, top_k=False, k=None):
    model = SentenceTransformer(model_path)
    abstracts = load_dict()['abstract']

    encoded_queries = model.encode(queries)
    if cache:
        json_path = model_path + '/abstract_embeddings.json'
        if os.path.exists(json_path):
            in_file = open(json_path, 'r')
            encoded_abstracts = [np.array(embed) for embed in json.load(in_file)]
            in_file.close()
        else:
            encoded_abstracts = model.encode(abstracts)
            out_file = open(json_path, 'w')
            json.dump([embed.tolist() for embed in encoded_abstracts], out_file)
            out_file.close()
    else:
        encoded_abstracts = model.encode(abstracts)

    ret = []  # best-matched abstracts for each query
    for encoded_query in encoded_queries:
        cosine_sim = [1-spatial.distance.cosine(encoded_query, a) for a in encoded_abstracts]
        idx = cosine_sim.index(max(cosine_sim))
        ret.append(abstracts[idx])
    return ret


if __name__ == '__main__':
    main()

    print(predict('output/finetuned2021-09-10_00-09-19', ['dosage primary antipeptic direct contact', 'regulatory mutants of Klebsiella aerogenes',
                                                          'Biochemical studies on camomile components/III. In vitro studies about the antipeptic activity of (--)-alpha-bisabolol (author\'s transl)']))

