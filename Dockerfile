FROM --platform=linux/amd64 pytorch/pytorch
# Use an appropriate base image for your algorithm.
# As an example, we use the official pytorch image.

# In reality this baseline algorithm only needs numpy so we could use a smaller image:
#FROM --platform=linux/amd64 python:3.11-slim


# Ensure that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y wget
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# Install requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt


#RUN wget -P /opt/app/resources https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user download.py /opt/app/

RUN python download.py
# Add any other files that are needed for your algorithm
# COPY --chown=user:user <source> <destination>

ENTRYPOINT ["python", "inference.py"]