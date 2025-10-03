# Dockerfile for running Planemo
# Build image using: docker build -t galaxy_planemo_no_root .
# Create container using: docker run -it -p 9090:9090 -v "/e/:/app/tools" galaxy_planemo_no_root
    # NOTE: Requires port 9090 to be opened (Planemo port)
# Inside container, call 'planemo serve' with host 0.0.0.0: planemo serve --galaxy_python_version 3.9 --host 0.0.0.0 /app/tools
    # NOTE: Exact Python 3.9 is required for the conda version of ImmunoPheno
# Wait until you see the "Serving on 0.0.0.0:9090" message in green and then go to "localhost:9090" on your local browser

# Use the stable Ubuntu 22.04 LTS base image
FROM ubuntu:22.04

# Set environment variable for non-interactive apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# Update apt-get and install all necessary packages
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        git \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        ca-certificates \
        sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Install Planemo using pip
RUN pip install planemo

# --- Create a non-root user for development ---
# Create a new user called 'galaxydev' with user ID 1001
RUN useradd -m -u 1001 -s /bin/bash galaxydev
# Give this new user passwordless sudo privileges (useful for debugging if needed)
RUN echo "galaxydev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# Change ownership of the working directory to our new user
RUN chown -R galaxydev:galaxydev /app

# --- Switch to the non-root user ---
# All subsequent commands in this Dockerfile and in the running container will be as 'galaxydev'
USER galaxydev

# Update PATH for the new user
ENV PATH="/home/galaxydev/.local/bin:${PATH}"

# Define a default command when the container starts (opens a bash shell)
CMD ["bash"]

# Define the default planemo port
EXPOSE 9090