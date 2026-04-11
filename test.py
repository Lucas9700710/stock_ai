from datetime import date, timedelta
print((date.today() - timedelta(days=1000)).strftime('%Y-%m-%d'))