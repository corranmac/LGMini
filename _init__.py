translatepy
deep-translator
timm
ruclip

from translatepy import Translator, Language
from deep_translator import DeeplTranslator, GoogleTranslator
from pathlib import Path
import csv
import logging
import requests
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from ruclip import load as get_ruclip
import multiprocessing
import torch
import os
from google.colab import output, drive
from psutil import virtual_memory
from pathlib import Path
from pynvml import *
import cv2
import urllib.request
import matplotlib.pyplot as plt
import numpy



ts = Translator()
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
if info.total>10252636672:
  print('Everything is ok, you can begin')
else:
  print('We dont recomend to begin, you gonna get out of memory')
