# testcontainer/Dockerfile

FROM python:3
WORKDIR /usr/src/niceuitools

# Copy files
COPY . .

# Install package
RUN pip install --no-cache-dir .

# Install dev dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt