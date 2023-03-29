# DDRX
FROM oubush/python_with_neper:latest
WORKDIR /app

COPY ./matgen/ ./matgen/
COPY ./visual_simul.py ./visual_simul.py
COPY ./visual_results.py ./visual_results.py
# RUN python -m pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 5006
CMD ["bokeh", "serve", "visual_simul.py", "visual_results.py"]
# RUN tar -xf v4.5.0.tar.gz
# WORKDIR /opt/neper-4.5.0/src/build
# RUN cmake .. && make -j && make installs