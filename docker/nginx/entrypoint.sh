#!/bin/sh

[ -z "$HOST" ] && HOST=127.0.0.1
[ -z "$PORT" ] && PORT=3000

sed "s/%HOST%/$HOST/g" -i /etc/nginx/conf.d/default.conf
sed "s/%PORT%/$PORT/g" -i /etc/nginx/conf.d/default.conf

/usr/sbin/nginx