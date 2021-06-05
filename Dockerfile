FROM nvcr.io/nvidia/pytorch:20.07-py3

RUN conda install cudatoolkit=11.0

CMD ["/bin/bash"]