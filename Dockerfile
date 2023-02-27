# DDRX
FROM python:3.11
WORKDIR /app

COPY ./build-image/ .
RUN chmod +x ./initial-setup.sh
RUN ./initial-setup.sh
# RUN neper -T -id 42 -n 42 -dim 2
CMD ["python"]
# RUN tar -xf v4.5.0.tar.gz
# WORKDIR /opt/neper-4.5.0/src/build
# RUN cmake .. && make -j && make installs