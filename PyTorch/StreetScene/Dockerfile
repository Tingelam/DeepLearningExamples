FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace

ENV PYTHONPATH "${PYTHONPATH}:/workspace"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk python3-pip git tmux htop tree libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy StreetScene code
COPY . .

# Default command
CMD ["bash"]