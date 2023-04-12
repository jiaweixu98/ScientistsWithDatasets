from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle as pk
from tqdm import tqdm
from collections import Counter
author = pd.read_csv('author.csv', index_col=0)
cluster_authors = {}
for i in range(5):
    cluster_authors[i] = list(map(str,list(author.loc[author['clusterID'] == i].index)))
cluster_authors_bioentity = {}
# 作者集合、论文集合
# author_set = set(list(map(str, list(author.index))))
# paper_set = set()

# f = open("../data/paper_author.csv", "r")
# author_paper = {}
# for line in tqdm(f):
#     line = line.strip().split(',')
#     if line[0] == 'PMID':
#         continue
#     if line[3] in author_paper:
#         author_paper[line[3]].append(line[0])
#     else:
#         author_paper[line[3]] = [line[0]]
# f.close()
# print('len(author_paper)',len(author_paper))
# pk.dump(author_paper,open('author_paper.pkl','wb'))


# f = open("../data/paper_bioentity.csv", "r")
# paper_bioentity = {}
# for line in tqdm(f):
#     line = line.strip().split(',')
#     if line[0] == 'PMID':
#         continue
#     elif line[0] in paper_bioentity:
#         paper_bioentity[line[0]].append(line[1])
#     else:
#         paper_bioentity[line[0]] = [line[1]]
# f.close()
# print('len(paper_bioentity)',len(paper_bioentity))
# pk.dump(paper_bioentity,open('paper_bioentity.pkl','wb'))
paper_bioentity = pk.load(open('paper_bioentity.pkl', 'rb'))
author_paper = pk.load(open('author_paper.pkl','rb'))
for k,v in tqdm(cluster_authors.items()):
    # 构建词组
    cluster_authors_bioentity[k] = []
    for author in v:
        # 对应的paper集合
        paper_set = author_paper[author]
        for paper in paper_set:
            try:
                cluster_authors_bioentity[k] += paper_bioentity[paper]
            except:
                continue
for k in range(5):
    print('len(cluster_authors_bioentity[k]', len(
        cluster_authors_bioentity[k]))

TOP_K_KEYWORDS = 100
tokenized_list_of_sentences = [
    cluster_authors_bioentity[0], 
    cluster_authors_bioentity[1],
    cluster_authors_bioentity[2],
    cluster_authors_bioentity[3],
    cluster_authors_bioentity[4]]


def identity_tokenizer(text):
    return text


vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
vectorizer.fit_transform(tokenized_list_of_sentences)
feature_names = vectorizer.get_feature_names()


def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def get_keywords(vectorizer, feature_names, doc):
    """Return top k keywords from a doc using TF-IDF method"""

    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])

    #sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    #extract only TOP_K_KEYWORDS
    keywords = extract_topn_from_vector(
        feature_names, sorted_items, TOP_K_KEYWORDS)

    return list(keywords.keys())


def extract_topn_from_vector(feature_names, sorted_items, topn=2):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


print('cluster_authors_bioentity[0]',get_keywords(vectorizer, feature_names, cluster_authors_bioentity[0]))

print('cluster_authors_bioentity[1]',get_keywords(vectorizer, feature_names, cluster_authors_bioentity[1]))

print('cluster_authors_bioentity[2]',get_keywords(vectorizer, feature_names, cluster_authors_bioentity[2]))

print('cluster_authors_bioentity[3]',get_keywords(vectorizer, feature_names, cluster_authors_bioentity[3]))

print('cluster_authors_bioentity[4]',get_keywords(vectorizer, feature_names, cluster_authors_bioentity[4]))