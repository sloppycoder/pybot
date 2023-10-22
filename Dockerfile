FROM python:3.10-bullseye as builder

run apt-get update && apt-get install -y \
    build-essential

COPY requirements.txt .
RUN pip install --root="/install" -r requirements.txt

# runtime
FROM python:3.10-slim-bullseye

COPY --from=builder /install /
COPY . .

CMD  ["/bin/sleep", "10000"]
