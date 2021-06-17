FROM nginx:1.19.8-alpine

COPY ./conf/conf.d /etc/nginx/conf.d
COPY ./conf/nginx.conf /etc/nginx/nginx.conf
COPY ./entrypoint.sh /entrypoint.sh

CMD ["/bin/sh", "/entrypoint.sh"]