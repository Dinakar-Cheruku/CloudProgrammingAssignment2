# Build stage
FROM python:3.9-slim as build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY . /app/
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install numpy pandas scikit-learn

# Final stage
FROM apache/spark-py:v3.3.2
WORKDIR /app
COPY --from=build /app /app
COPY src/ValidationDataset.csv /app/
COPY src/winemodel /app/winemodel/
CMD ["spark-submit", "src/main.py", "ValidationDataset.csv", "winemodel"]

