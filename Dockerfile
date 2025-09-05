FROM python:3.11-slim-trixie
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "main.py"]