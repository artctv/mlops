from airflow import DAG
import airflow
from datetime import datetime
from airflow.operators.python import PythonOperator


def test123():
    print("Hello world!")



args = {
    'owner': 'admin',
    'start_date':datetime(2018, 11, 1),
    'provide_context':True
}

dag = DAG(
    'Hello-world_example',
    description='Hello-world example',
    schedule_interval='*/1 * * * *',
    catchup=False,
    default_args=args
)
# dag
    # task_1 = PythonOperator(task_id="task_1", python_callable=test123)