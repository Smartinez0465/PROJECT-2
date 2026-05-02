FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_analysis.py .

RUN mkdir -p /output

CMD ["python", "run_analysis.py"]
