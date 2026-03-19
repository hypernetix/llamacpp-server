FROM python:3.12-slim

WORKDIR /app

COPY tests/openai-compat/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tests/openai-compat/test_openai_compat.py .

ENTRYPOINT ["python", "test_openai_compat.py"]
