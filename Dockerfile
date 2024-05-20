# ビルダーイメージ
ARG PYTHON_VERSION=3.10-slim
FROM python:${PYTHON_VERSION} as builder

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl linux-libc-dev libc6-dev

# PATHの設定
ENV PATH /root/.local/bin:$PATH
WORKDIR /app

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python3 -

# キャッシュの最適化のため、プロジェクトの依存関係ファイルをコピー
COPY ./pyproject.toml ./poetry.lock* /app/

# Poetryの設定と依存関係のインストール
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# 本番用イメージ
FROM python:${PYTHON_VERSION} as runner

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libzmq3-dev \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ビルダーからPythonパッケージをコピー
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Jupyterのインストール
# RUN pip install jupyter

# 作業ディレクトリの設定
WORKDIR /app

# Jupyter Notebookを起動
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
