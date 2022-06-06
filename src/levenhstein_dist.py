from leven import levenshtein

def get_leven_dist(words1: list(), words2: list()) -> list():
    """
    get levenshtein distance of 2 lists of words
    :param words1: first word list
    :param words2: second word list
    :return: list of levenshtein distances
    """

    res = list()

    for i in range(len(words1)):
        lv = levenshtein(words1[i], words2[i])
        res.append(lv)

    return res
