# DDRX
FROM python:3.11
WORKDIR /app

COPY . .
# COPY ./build-image/ ./build-image/
RUN chmod +x ./build-image/initial-setup.sh
RUN ./build-image/initial-setup.sh
# RUN neper -T -id 42 -n 42 -dim 2

# COPY ./requirements.txt ./requirements.txt
# COPY . .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 5006
CMD ["bokeh", "serve", "visual_test.py"]
# RUN tar -xf v4.5.0.tar.gz
# WORKDIR /opt/neper-4.5.0/src/build
# RUN cmake .. && make -j && make installs