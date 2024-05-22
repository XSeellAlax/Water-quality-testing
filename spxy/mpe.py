import numpy as np
from itertools import permutations

def MPE(data, n, m, r):
    # &quot;&quot;&quot;
    # :param data: 时间序列数据
    # :param n: 分析的尺度数
    # :param m: 分析的子序列长度
    # :param r: 相似度判定的阈值
    # :return: 多尺度排列熵值
    # &quot;&quot;&quot;
    def permutation_entropy(data, m, r):
        data_len = len(data)
        permu = list(permutations(range(m)))
        freq = np.zeros(len(permu))
        for i in range(data_len-m+1):
            sorted_idx = np.argsort(data[i:i+m])
            for j in range(len(permu)):
                if all(np.array(permu[j]) == sorted_idx):
                    freq[j] += 1
                    break
        prob = freq / (data_len - m + 1)
        pe = -sum(prob * np.log(prob))
        return pe
    
    pe = np.zeros(n)
    for i in range(n):
        t = int(len(data) / (i+1))
        sub_data = [data[j*t:(j+1)*t] for j in range(i+1)]
        pe[i] = sum([permutation_entropy(d, m, r) for d in sub_data]) / (i+1)
    return pe

data = np.random.rand(30)
pe = MPE(data, 5, 2, 0.1)
print(pe)