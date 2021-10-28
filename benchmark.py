from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import cosine

from preproc import load_dict, load_input_examples
from finetune import predict


def tf_idf_predict(queries, top_k=False, k=None):
    abstracts = load_dict()['abstract']
    corpus = queries + abstracts

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform(corpus).toarray()
    # print(type(tfidf_vectors))
    # print(tfidf_vectors.shape)

    ret = []
    for i in range(len(queries)):
        cosine_sim = [1-cosine(tfidf_vectors[i], tfidf_vectors[j]) for j in range(len(queries), len(corpus))]
        idx = cosine_sim.index(max(cosine_sim))
        ret.append(corpus[len(queries) + idx])
    return ret


def main():
    test_examples = [i for i in load_input_examples(cached=True)[25000:27000] if i.label > 0.5]
    titles = [i.texts[0] for i in test_examples]
    abstracts = [i.texts[1] for i in test_examples]

    pred = tf_idf_predict(titles)
    count = 0
    for i in range(len(pred)):
        if abstracts[i] == pred[i]:
            count += 1
    tf_idf_acc = count / len(pred)
    print(f'TF-IDF accuracy: {round(tf_idf_acc, 4)}')

    pred = predict('output/finetuned2021-09-10_00-09-19', titles)
    count = 0
    for i in range(len(pred)):
        if abstracts[i] == pred[i]:
            count += 1
    sbert_acc = count / len(pred)
    print(f'sbert accuracy: {round(sbert_acc, 4)}')


if __name__ == '__main__':
    # print(tf_idf_predict(['dosage primary antipeptic direct contact', 'regulatory mutants of Klebsiella aerogenes',
    #                      'Biochemical studies on camomile components/III. In vitro studies about the antipeptic activity of (--)-alpha-bisabolol (author\'s transl)']))
    main()
