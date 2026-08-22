FROM python:3.11-slim

# --- system dependencies -----------------------------------------------
# build-essential: needed to compile the visilibity C++ extension
# the lib* packages: needed at runtime for PyQt5 (Qt "xcb" platform plugin)
# and pygame (SDL2) to talk to an X11 display forwarded from the host
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    libxext6 \
    libxrender1 \
    libxi6 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-xfixes0 \
    libxcb1 \
    libx11-xcb1 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached independently
# of the application source, which is bind-mounted in at `docker run` time
# rather than copied into the image (see docker-run.sh).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
