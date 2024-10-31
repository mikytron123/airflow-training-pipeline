#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE airflow;
    CREATE USER airflow_user WITH PASSWORD 'airflow_pass';
    GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow_user;
    -- PostgreSQL 15 requires additional privileges:
    GRANT ALL ON SCHEMA public TO airflow_user;
    ALTER DATABASE airflow OWNER TO airflow_user;

    CREATE DATABASE mlflow;
    CREATE USER mlflow_user WITH PASSWORD 'mlflow_pass';
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;
    -- PostgreSQL 15 requires additional privileges:
    GRANT ALL ON SCHEMA public TO mlflow_user;
    ALTER DATABASE mlflow OWNER TO mlflow_user;
EOSQL
    