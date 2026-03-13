FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p models results data
EXPOSE 8501
CMD ["sh", "-c", "python src/train_random_forest.py && streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0"]