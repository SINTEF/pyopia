FROM python:3.11-buster

WORKDIR /src

RUN pip install poetry

COPY . .

RUN poetry install

ENTRYPOINT ["tail", "-f", "/dev/null"]
