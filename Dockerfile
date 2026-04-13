FROM python:3.11-slim

# 1. Set the working directory
WORKDIR /app

# 2. Create a non-root user (Hugging Face standard)
RUN useradd -m -u 1000 user

# 3. Copy your project files into the container
# We do this as the root user first so we can set permissions
COPY . /app

# 4. FIX PERMISSIONS: Give the 'user' ownership of the /app folder
# This allows the app to create 'document_files' and other folders
RUN chown -R user:user /app && chmod -R 777 /app

# 5. Switch to the non-root user
USER user

# 6. Set up the Python path and environment
ENV PYTHONPATH=/app
ENV PATH="/home/user/.local/bin:${PATH}"

# 7. Install dependencies as the 'user'
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. Expose the port Hugging Face expects
EXPOSE 7860

# 9. Start the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]