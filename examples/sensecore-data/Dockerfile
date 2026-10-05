FROM registry.cn-sh-01.sensecore.cn/ccr-zhicheng-02/elf@sha256:f6b963095ebf1aa4b516ff8a958ba3d0392eac9b3f85de072c3ab766b30420fb
USER root
RUN python3 -m venv --without-pip --system-site-packages /opt/user-env
ENV PATH="/opt/user-env/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir colorama==0.4.6
RUN python3 -c 'import json, pathlib, colorama; pathlib.Path("/usr/local/share/user-dockerfile-proof.json").write_text(json.dumps({"origin":"client-dockerfile", "colorama":colorama.__version__}))'
