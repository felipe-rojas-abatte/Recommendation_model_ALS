import numpy as np
from numpy import linalg
from numpy.linalg import norm

import pandas as pd
from typing import Optional, List, Tuple
from datetime import datetime
from datetime import date
import random
import scipy.sparse as sparse
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import spsolve

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

import sys
import math
import shutil
import implicit 
from implicit import evaluation

import time
import os
from os import listdir
from os.path import isfile, isdir
import google
from google.cloud.bigquery import Client
from google.cloud.bigquery import QueryJobConfig
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import logging

client = Client()
current_dir = os.getcwd()