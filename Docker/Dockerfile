FROM pytorch/pytorch
LABEL maintainer="durgesh1993"	

RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get -y install libsndfile1
RUN pip install pandas
RUN pip install noisereduce
RUN pip install opencv-python-headless
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install wandb
RUN pip install tabulate
RUN pip install intervaltree
RUN pip install soundfile
RUN pip install librosa
RUN pip install awscli
RUN useradd -ms /bin/bash durgesh
USER durgesh
