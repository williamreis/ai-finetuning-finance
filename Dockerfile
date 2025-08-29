FROM python:3.10-slim

# Instalar dependências básicas
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Criar diretório de trabalho
WORKDIR /app
COPY . /app

# Comando padrão: treino
CMD ["python", "train.py"]
