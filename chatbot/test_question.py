import os
import traceback

os.chdir(r'E:\B. Data Analyst\2. Bootcamp\Project-5 ( Jio_Subscriber_Retention_Churn)\2. Clean Data\Project_5\chatbot')

try:
    import app
    question = 'What is the churn rate by circle?'
    print('QUESTION:', question)
    output = app.run_sql_answer(question)
    print('SQL:')
    print(output['sql'])
    print('ROWS:', len(output['result']))
    print(output['result'].head().to_string(index=False))
except Exception:
    traceback.print_exc()
    raise
