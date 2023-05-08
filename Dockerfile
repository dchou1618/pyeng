FROM python:3.9

COPY Modeling/Regressions:Classifications/causal.py /app/

WORKDIR /app

CMD ["python", "causal.py"]
