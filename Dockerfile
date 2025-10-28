# Dockerfile para Singular Finance
FROM python:3.10-slim

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Instalar biblioteca em modo de desenvolvimento
RUN pip install -e .

# Expor porta (se necessário para interface web futura)
EXPOSE 8000

# Comando padrão
CMD ["python", "-c", "import singular_finance; print('Singular Finance instalado com sucesso!')"]
