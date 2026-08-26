# Packages the FastAPI backend. The worker uses this SAME image (see
# render.yaml's startCommand override) so both processes always run
# identical code — there's only one Dockerfile to maintain.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Overridden for the worker service — see render.yaml.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
