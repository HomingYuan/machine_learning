from nltk.tokenize import sent_tokenize, word_tokenize # nltk.tokenize是NLTK提供的分词工具包
# sent_tokenize() 函数对应的是分段为句。 word_tokenize()函数对应的是分句为词。
from nltk.corpus import stopwords  # stopwords 是一个列表，包含了英文中那些频繁出现的词，如am, is, are
from collections import defaultdict  # defaultdict 是一个带有默认值的字典容器
from string import punctuation  # puctuation 是一个列表，包含了英文中的标点和符号
from heapq import nlargest  # nlargest() 函数可以很快地求出一个容器中最大的n个数字




stopwords = set(stopwords.words('english') + list(punctuation))
max_cut = 0.9
min_cut = 0.1

def compute_frequencies(word_sent):
    """
    计算出每个词出现的频率
    :param word_sent: 是一个已经分好词的列表
    :return: 一个词典freq[], freq[w]代表了w出现的频率
    """

    freq = defaultdict(int)
    for s in word_sent:
        for word in s:
            if word not in stopwords:
                freq[word] += 1

    m = float(max(freq.values()))
    for w in list(freq.keys()):
        freq[w] = freq[w]/m
        if freq[w] >= max_cut or freq[w] <= min_cut:
            del freq[w]
    return freq

def summarize(text, n):
    """
    用来总结的主要函数
    :param text:
    :param n:
    :return:
    """

    # 首先先把句子分出来
    sents = sent_tokenize(text)
    assert n <= len(sents)

    # 然后再分词
    word_sent = [word_tokenize(s.lower()) for s in sents]
    # self._freq是一个词和词频率的字典
    freq = compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i, word in enumerate(word_sent):
        for w in word:
            if w in freq:
                ranking[i] += freq[w]
    sents_idx = rank(ranking, n)
    return [sents[j] for j in sents_idx]


def rank(ranking, n):
    return nlargest(n, ranking, key=ranking.get)


if __name__ == '__main__':
    with open("news.txt", "r") as myfile:
        text = myfile.read().replace('\n','')
    res = summarize(text, 2)
    for i in range(len(res)):
        print("* " + res[i])