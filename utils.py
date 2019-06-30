class HistoryMeter:
    def __init__(self):
        self.rouge = {
            'rouge-l': {'p': [], 'r': [], 'f': []},
            'rouge-1': {'p': [], 'r': [], 'f': []},
            'rouge-2': {'p': [], 'r': [], 'f': []},

        }
        self.time = []
        self.loss = []
        self.n = 0

    def inc_count(self, n=1):
        self.n += n

    def update(self, metric, values):
        if metric == 'loss':
            self.loss.append(values)
        elif metric == 'time':
            self.time.append(values)
        elif metric.startswith('rouge'):
            for key, value in values.items():
                self.rouge[metric][key].append(value)
        else:
            raise Exception(f'Unkwown metric {metric}')


class AvgMeter:
    def __init__(self):
        self.rouge = {
            'rouge-l': {'p': 0., 'r': 0., 'f': 0.},
            'rouge-1': {'p': 0., 'r': 0., 'f': 0.},
            'rouge-2': {'p': 0., 'r': 0., 'f': 0.},

        }
        self.loss = 0.
        self.n = 0

    def inc_count(self, n=1):
        self.n += n

    def get_avgs(self):
        return {
            'loss': self.loss / self.n,
            'rouge-l': {k: v / self.n for k, v in self.rouge['rouge-l'].items()},
            'rouge-1': {k: v / self.n for k, v in self.rouge['rouge-1'].items()},
            'rouge-2': {k: v / self.n for k, v in self.rouge['rouge-2'].items()},
        }

    def update(self, metric, values):
        if metric == 'loss':
            self.loss += values
        elif metric.startswith('rouge'):
            for key, value in values.items():
                self.rouge[metric][key] += value
        else:
            raise Exception(f'Unkwown metric {metric}')
