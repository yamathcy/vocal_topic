import numpy as np

from collections import Counter

def sampling_words(topic_num, alpha_param, beta_param):
    alpha = np.ones(topic_num)
    topic_dist = 