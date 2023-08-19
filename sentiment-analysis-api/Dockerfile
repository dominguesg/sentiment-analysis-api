FROM python:3.8-slim-buster
WORKDIR /api
#ENV TZ=America/sao_paulo

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 8000
RUN mkdir -p ./assets

# CMD ["bin/start_server"]
# ENTRYPOINT [ "start_server.sh" ]

COPY start_server.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start_server.sh
CMD ["/usr/local/bin/start_server.sh"]