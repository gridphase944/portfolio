FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3


# ===== 基本ツール =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ（container内のpath。host directoryはmountしない）
WORKDIR /workspace

# ===== Python環境 =====
# TensorFlowはNGC内蔵版を使う。historicalのversionへ上書きしない。
# その他の依存は公開requirementsのpin/rangeを保持する。
COPY requirements.txt ./requirements.txt
RUN sed '/^tensorflow==/d' requirements.txt > /tmp/i12-python-requirements.txt \
    && python3 -m pip install --no-cache-dir -r /tmp/i12-python-requirements.txt \
    && rm /tmp/i12-python-requirements.txt

# 実versionはbuild logで確認する。I12 historical runとの一致は保証しない。
RUN python3 -c "import json, platform, numpy, pandas, tensorflow as tf; print(json.dumps({'python': platform.python_version(), 'tensorflow': tf.__version__, 'numpy': numpy.__version__, 'pandas': pandas.__version__}))"

# ===== 便利設定 =====
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# ===== Public runtime / focused tests =====
# datasetと実行artifactはCOPYせず、実行時に外部pathを指定する。
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY tests/ ./tests/

# ===== デフォルト起動 =====
# python3 scripts/run_experiment.py --help
CMD ["bash"]
