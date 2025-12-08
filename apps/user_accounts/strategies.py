class ChartStrategy:
    def render(self, labels, data, label, color):
        raise NotImplementedError

class LineChart(ChartStrategy):
    def render(self, labels, data, label, color):
        return {
            'type': 'line',
            'labels': labels,
            'data': data,
            'label': label,
            'color': color
        }

class BarChart(ChartStrategy):
    def render(self, labels, data, label, color):
        return {
            'type': 'bar',
            'labels': labels,
            'data': data,
            'label': label,
            'color': color
        }


