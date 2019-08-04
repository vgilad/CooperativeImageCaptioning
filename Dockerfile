FROM nvcr.io/nvidia/pytorch:18.09-py3
RUN apt-get update && apt-get install -y bc
RUN conda install -y pandas=0.23.0
RUN conda install -y h5py=2.8.0
#RUN conda install -y tensorflow=1.8.0
RUN conda install -y scikit-image=0.13.1

# Install OpenJDK-8
RUN apt-get update && \
apt-get install -y openjdk-8-jdk && \
apt-get install -y ant && \
apt-get clean;

# Fix certificate issues
RUN apt-get update && \
apt-get install ca-certificates-java && \
apt-get clean && \
update-ca-certificates -f;
# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

WORKDIR /project

