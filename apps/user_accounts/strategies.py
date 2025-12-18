class ChartStrategy:
    def render(self, labels, data, label, color):
        raise NotImplementedError

import json

class LineChart(ChartStrategy):
    def render(self, labels, data, label, color):
        return {
            'type': 'line',
            'labels': json.dumps(labels), 
            'data': json.dumps(data),     
            'label': label,
            'color': color
        }

class BarChart(ChartStrategy):
    def render(self, labels, data, label, color):
        return {
            'type': 'bar',
            'labels': json.dumps(labels),  # Add this
            'data': json.dumps(data),      # Add this
            'label': label,
            'color': color
        }