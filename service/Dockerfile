FROM python:3.7-slim

# RUN mkdir app
# WORKDIR /app
COPY main.py .
COPY requirements.txt .
COPY model.joblib .

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]