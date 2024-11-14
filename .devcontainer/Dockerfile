FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

RUN sudo apt --fix-broken install
RUN sudo apt-get update -y
RUN sudo apt install -y chromium
RUN sudo apt install -y chromium-driver
RUN wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
RUN unzip chromedriver_linux64.zip
