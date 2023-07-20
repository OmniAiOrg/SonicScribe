import pypinyin

def match_characters(ground_truth, predicted):
    matches = []
    n = len(ground_truth)
    m = len(predicted)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ground_truth[i - 1] == predicted[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 4
            else:
                gt_pinyin = pypinyin.lazy_pinyin(ground_truth[i - 1])[0]
                pd_pinyin = pypinyin.lazy_pinyin(predicted[j - 1])[0]
                if gt_pinyin == pd_pinyin:
                    dp[i][j] = dp[i - 1][j - 1] + 2
                elif gt_pinyin.startswith(pd_pinyin) or pd_pinyin.startswith(gt_pinyin):
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    i = n
    j = m
    while i > 0 and j > 0:
        if ground_truth[i - 1] == predicted[j - 1]:
            matches.append((ground_truth[i - 1], j - 1))
            i -= 1
            j -= 1
        else:
            gt_pinyin = pypinyin.lazy_pinyin(ground_truth[i - 1])[0]
            pd_pinyin = pypinyin.lazy_pinyin(predicted[j - 1])[0]
            if gt_pinyin == pd_pinyin:
                matches.append((ground_truth[i - 1], j - 1))
                i -= 1
                j -= 1
            elif gt_pinyin.startswith(pd_pinyin) or pd_pinyin.startswith(gt_pinyin):
                matches.append((ground_truth[i - 1], j - 1))
                i -= 1
                j -= 1
            elif dp[i][j - 1] > dp[i - 1][j]:
                j -= 1
            else:
                i -= 1
    matches.reverse()
    return matches

if __name__ == '__main__':
    print(
        match_characters(
            [i for i in '你是一个好人'],
            [i for i in '可是尼是亿个好的热物']
        )
    )