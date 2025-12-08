from .models import VitalStats
from datetime import datetime

class ChartDataFactory:
    def __init__(self, user_id):
        self.user_id = user_id

    def get_data(self, chart_type):
        labels = []
        data = []

        if chart_type == 'bp':
            # Django ORM: using user__id to filter by foreign key relationship
            records = VitalStats.objects.filter(user__id=self.user_id).order_by('-date')[:7]
            for r in reversed(records):
                labels.append(r.date.strftime("%a"))
                data.append(r.bp_systolic)

        elif chart_type == 'sugar':
            # Django ORM: using user__id to filter by foreign key relationship
            records = VitalStats.objects.filter(user__id=self.user_id).order_by('-date')[:7]
            print("Sugar Records:", records)
            for r in reversed(records):
                labels.append(r.date.strftime("%a"))
                if r.sugar_level is not None:
                    data.append(r.sugar_level)
                else:
                    data.append(0)  # 0 daal diya agar koi value missing ho

        print("Labels:", labels)
        print("Data:", data)

        return labels, data