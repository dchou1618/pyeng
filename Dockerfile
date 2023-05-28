FROM python:3.9

COPY Modeling/Regressions:Classifications/causal.py /app/

COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "causal.py"]
