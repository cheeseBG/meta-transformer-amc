FROM python:3.9.7

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "main.py", "test"]

