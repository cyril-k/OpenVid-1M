FROM nvcr.io/nvidia/pytorch:24.05-py3

# COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /uvx /bin/
ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

# RUN uv sync --frozen

# RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
RUN echo "Installing Apex on top of ${BASE_IMAGE}"
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

WORKDIR /workspace

RUN pip uninstall -y flash-attn || :
RUN pip install flash-attn --no-build-isolation

RUN apt update && apt install -y ffmpeg
RUN pip install torchcodec==0.0.3

ADD . /workspace

ENV PYTHONPATH=/workspace:$PYTHONPATH