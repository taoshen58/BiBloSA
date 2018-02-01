def do_analyse_trec(file_path, use_loss=False, stop=None):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        output = [0, 0., 0.]  # xx, dev, test,
        for line in file:
            if line.startswith('data round'):  # get step
                output[0] = int(line.split(' ')[-4].split(':')[-1])
                if stop is not None and output[0] > stop: break
            if line.startswith('==> for dev'):  # dev
                output[1] = float(line.split(' ')[-1])  # accu
                output[2] = float(line.split(' ')[-3][:-1])  # loss
                results.append(output)
                output = [0, 0., 0.]

    # max step
    if len(results) > 0:
        print('max step:', results[-1][0])

    # sort
    sort = 1 if not use_loss else 2
    output = list(sorted(results, key=lambda elem: elem[sort], reverse=not use_loss))

    for elem in output[:20]:
        print('step: %d, dev: %.4f, dev_loss: %.4f' %
              (elem[0], elem[1], elem[2]))


if __name__ == '__main__':
    file_path = '/Users/xxxx/Desktop/tmp/file_transfer/qc/Oct--9-01-57-54_log.txt'
    use_loss = False
    do_analyse_trec(file_path, use_loss, None)


