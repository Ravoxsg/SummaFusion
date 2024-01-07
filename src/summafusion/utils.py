import numpy as np


def rank_array(t):
    y = np.copy(t)
    y.sort()
    y = y[::-1]
    ranks = np.zeros(len(t))
    flagged = np.zeros(len(t))
    for i in range(len(t)):
        el = t[i]
        for j in range(len(t)):
            if el == y[j] and flagged[j] == 0:
                ranks[i] = j
                flagged[j] = 1
                break

    return ranks


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)

    return tensors.detach()


def unique_idx(t):
    if len(t.shape) == 2:
        reduced_t = t.sum(axis = 0)
    else:
        reduced_t = t
    idx = []
    items = []
    for i in range(len(reduced_t)):
        if not(reduced_t[i].item() in items):
            items.append(reduced_t[i].item())
            idx.append(i)
    idx = np.array(idx)
    p = np.random.permutation(len(idx))
    idx = idx[p]
    idx = list(idx)

    return idx


def prune_idx(scores, args):
    s = scores.detach().cpu().numpy()
    if args.pos_neg_construction in ["overall_sum_mean", "overall_sum"]:
        if len(s.shape) == 2:
            reduced_s = np.sum(s, 0)
        else:
            reduced_s = s
        sort_idx = np.argsort(reduced_s)[::-1]
        if args.sampling_strat == "bottom":
            neg = list(sort_idx[-args.n_negatives:])
        elif args.sampling_strat == "random":
            p = np.random.permutation(len(sort_idx) - args.n_positives)
            neg = list(sort_idx[args.n_positives:][p][:args.n_negatives])
        idx_to_keep = list(sort_idx)[:args.n_positives] + neg
    elif args.pos_neg_construction == "per_task":
        idx_to_keep = []
        for j in range(args.n_tasks):
            scores_j = s[j]
            sort_idx = np.argsort(scores_j)[::-1]
            if args.sampling_strat == "bottom":
                neg = list(sort_idx[-args.n_negatives:])
            elif args.sampling_strat == "random":
                p = np.random.permutation(len(sort_idx) - args.n_positives)
                neg = list(sort_idx[args.n_positives:][p][:args.n_negatives])
            idx_to_keep_j = list(sort_idx)[:args.n_positives] + neg
            for idx in idx_to_keep_j:
                if not(idx in idx_to_keep):
                    idx_to_keep.append(idx)
    idx_to_keep = np.array(idx_to_keep)
    p = np.random.permutation(len(idx_to_keep))
    idx_to_keep = idx_to_keep[p]
    idx_to_keep = list(idx_to_keep)

    return idx_to_keep


if __name__ == '__main__':
    t = np.random.randint(0, 5, 10)
    print(t)
    ranks = rank_array(t)
    print(ranks)


