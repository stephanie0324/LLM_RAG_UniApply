#!/bin/bash


cd ${PWD}/..

docker run -it --rm -v ${PWD}:/app -p 1902:7860 -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt -e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt -v /etc/ssl/certs:/etc/ssl/certs python:3.9.18-bullseye bash
# docker run -it --rm -v ${PWD}:/app -p 1801:7860 python:3.9.18-bullseye bash
#docker run -it --rm -v ${PWD}:/app -p 1690:7860 --entrypoint /bin/bash itri/industry_demo:20231125
