# DDRX
FROM oubush/python_with_neper:latest
WORKDIR /app

COPY ./matgen/ ./matgen/
COPY ./visual_test.py ./visual_test.py
# RUN python -m pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 5006
CMD ["bokeh", "serve", "visual_test.py"]
# RUN tar -xf v4.5.0.tar.gz
# WORKDIR /opt/neper-4.5.0/src/build
# RUN cmake .. && make -j && make installs