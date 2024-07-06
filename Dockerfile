FROM python:3.9-slim

COPY . .

RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["app.py"]
ENTRYPOINT ["python"]