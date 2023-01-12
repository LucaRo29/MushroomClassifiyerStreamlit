import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torchvision import transforms
from captum.attr import GradientShap, Occlusion
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f'{ROOT_PATH}\\model_files\\shroom_ai.pt'

CLASSES = (
    'Agaricus',
    'Amanita',
    'Boletus',
    'Cortinarius',
    'Entoloma',
    'Hygrocybe',
    'Lactarius',
    'Russula',
    'Suillus')
num_classes = len(CLASSES)
