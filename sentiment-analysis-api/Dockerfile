FROM python:3.10.12
WORKDIR /api

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 8000
RUN mkdir -p ./assets

COPY start_server.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start_server.sh
CMD ["/usr/local/bin/start_server.sh"]