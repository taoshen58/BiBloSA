def do_analyse_sick(file_path, dev=True, delta=1, stop=None):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        find_entry = False
        output = [0, 0., 0., 0., 0., 0., 0., 0., 0.] # step, dev(loss, pearson, spearman, mse), \
                                                        # test(loss, pearson, spearman, mse),
        for line in file:
            if not find_entry:
                if line.startswith('data round'):  # get step
                    output[0] = int(line.split(' ')[-4].split(':')[-1])
                    if stop is not None and output[0] > stop: break
                if line.startswith('==> for dev'):  # dev
                    output[1] = float(line.split(' ')[-7][:-1])
                    output[2] = float(line.split(' ')[-5][:-1])
                    output[3] = float(line.split(' ')[-3][:-1])
                    output[4] = float(line.split(' ')[-1])
                    find_entry = True
            else:
                if line.startswith('~~> for test'):  # test
                    output[5] = float(line.split(' ')[-7][:-1])
                    output[6] = float(line.split(' ')[-5][:-1])
                    output[7] = float(line.split(' ')[-3][:-1])
                    output[8] = float(line.split(' ')[-1])
                    results.append(output)
                    find_entry = False
                    output = [0, 0., 0., 0., 0., 0., 0., 0., 0.]

    # max step
    if len(results) > 0:
        print('max step:', results[-1][0])

    # sort
    sort = 1 if dev else 5
    sort += delta
    output = list(sorted(results, key=lambda elem: elem[sort], reverse=delta in [1, 2]))

    for elem in output[:10]:
        print('step: %d, dev_loss: %.4f, dev_pearson: %.4f, dev_spm: %.4f, dev_mse: %.4f,'
              ' test_loss: %.4f, test_pearson: %.4f, test_spm: %.4f, test_mse: %.4f,' %
              (elem[0], elem[1], elem[2], elem[3],elem[4], elem[5], elem[6], elem[7],elem[8]))


if __name__ == '__main__':

    file_path = '/Users/xxxx/Desktop/tmp/file_transfer/sick/Oct--9-14-09-34_log.txt'
    dev = True
    use_loss = True

    do_analyse_sick(file_path, dev, 1, None)