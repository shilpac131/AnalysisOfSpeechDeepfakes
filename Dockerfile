FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Create the conda environment
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml

# Activate the environment
RUN echo "source activate aasist" > ~/.bashrc
ENV PATH /opt/conda/envs/aasist/bin:$PATH

# Command to run your application
CMD ["python", "main.py", "--eval", "--config", "./config/AASIST.conf"]
#-----------------------------------------------------------------------------------------------------


