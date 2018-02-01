import os


def do_analyse_snli(file_path, dev=True, use_loss=False, stop=None):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        find_entry = False
        output = [0, 0., 0., 0., 0.] # xx, dev, test,
        for line in file:
            if not find_entry:
                if line.startswith('data round'):  # get step
                    output[0] = int(line.split(' ')[-4].split(':')[-1])
                    if stop is not None and output[0] > stop: break
                if line.startswith('==> for dev'):  # dev
                    output[1] = float(line.split(' ')[-1])
                    output[2] = float(line.split(' ')[-3][:-1])
                    find_entry = True
            else:
                if line.startswith('~~> for test'):  # test
                    output[3] = float(line.split(' ')[-1])
                    output[4] = float(line.split(' ')[-3][:-1])
                    results.append(output)
                    find_entry = False
                    output = [0, 0., 0., 0., 0.]

    # max step
    if len(results) > 0:
        print('max step:', results[-1][0])

    # sort
    sort = 1 if dev else 3
    if use_loss: sort += 1
    output = list(sorted(results, key=lambda elem: elem[sort], reverse=not use_loss))

    for elem in output[:20]:
        print('step: %d, dev: %.4f, dev_loss: %.4f, test: %.4f, test_loss: %.4f' %
              (elem[0], elem[1], elem[2], elem[3],elem[4]))


if __name__ == '__main__':

    type = 0
    file_path = '/Users/xxxx/Desktop/tmp/file_transfer/snli/Oct--5-10-39-49_log.txt'
    dev = True
    use_loss = False

    do_analyse_snli(file_path, dev, use_loss, None)

