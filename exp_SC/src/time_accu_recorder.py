import os


class TimeAccuRecorder():
    def __init__(self, dataset_type, val_index, answer_dir):
        self.file_path = os.path.join(answer_dir, "time_vs_accu_data_%s_idx_%d.txt"%(dataset_type, val_index))
        self.data = [] # (time, accuracy)

    def add_data(self, time, val_accu):
        self.data.append((time, val_accu))

    def save_to_file(self):
        with open(self.file_path, 'w', encoding='utf-8') as file:
            time_base = 0
            for idx_d, d in enumerate(self.data):
                if idx_d == 0:
                    time_base = d[0]
                file.write('%.6f\t%.6f%s' % (d[0] - time_base, d[1], os.linesep))

