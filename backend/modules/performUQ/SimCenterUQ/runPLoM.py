# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Leland Stanford Junior University
# Copyright (c) 2021 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# This module is modified from the surrogateBuild.py to use PLoM for surrogate
# modeling while maintaining similar input/output formats compatable with the current workflow
#
# Contributors:
# Kuanshi Zhong
# Sang-ri Yi
# Frank Mckenna
#
from re import T
import time
import shutil
import os
import sys
import subprocess
import math
import pickle
import glob
import json
from scipy.stats import lognorm, norm
import numpy as np
import GPy as GPy

from pyDOE import lhs
import warnings
import random

from multiprocessing import Pool

from PLoM.PLoM import *
import pandas as pd

# ==========================================================================================

class runPLoM:

    def __init__(self, work_dir, run_type, os_type, inp, errlog):
        self.errlog = errlog
        self.work_dir = work_dir
        self.os_type = os_type
        self.run_type = run_type

        # reading X and Y variable names
        rv_name = list()
        self.g_name = list()
        x_dim = 0
        y_dim = 0
        for rv in inp['randomVariables']:
            rv_name = rv_name + [rv['name']]
            x_dim += 1
        if x_dim == 0:
            msg = 'Error reading json: RV is empty'
            errlog.exit(msg)
        for g in inp['EDP']:
            if g['length']==1: # scalar
                self.g_name = self.g_name + [g['name']]
                y_dim += 1
            else: # vector
                for nl in range(g['length']):
                    self.g_name = self.g_name + ["{}_{}".format(g['name'],nl+1)]
                    y_dim += 1
        if y_dim == 0:
            msg = 'Error reading json: EDP(QoI) is empty'
            errlog.exit(msg)

        self.id_sim = 0
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.rv_name = rv_name

        surrogateInfo = inp["UQ_Method"]["surrogateMethodInfo"]

        # parallel setup
        try:
            self.do_parallel = surrogateInfo["parallelExecution"]
        except:
            self.do_parallel = True
        if self.do_parallel:
            if self.run_type.lower() == 'runninglocal':
                self.n_processor = os.cpu_count()
                from multiprocessing import Pool
                self.pool = Pool(self.n_processor)
            else:
                from mpi4py import MPI
                from mpi4py.futures import MPIPoolExecutor
                self.world = MPI.COMM_WORLD
                self.pool = MPIPoolExecutor()
                self.n_processor = self.world.Get_size()
            print("nprocessor :")
            print(self.n_processor)
            self.cal_interval = self.n_processor
        else:
            self.pool = 0
            self.cal_interval = 5

        # prepare training data
        if surrogateInfo["method"] == "Import Data File":
            do_sampling = False
            do_simulation = not surrogateInfo["outputData"]
            self.doe_method = "None"  # default
            do_doe = False
            # self.inpData = surrogateInfo['inpFile']
            self.inpData = os.path.join(work_dir, "templatedir/inpFile.in")
            if not do_simulation:
                # self.outData = surrogateInfo['outFile']
                self.outData = os.path.join(work_dir, "templatedir/outFile.in")
        else:
            msg = 'Error reading json: only supporting "Import Data File"'
            errlog.exit(msg)

        if do_sampling:
            pass
        else:
            X = read_file(self.inpData,errlog)
            if len(X.columns) != x_dim:
                msg = 'Error importing input data: Number of dimension inconsistent: have {} RV(s) but {} column(s).' \
                    .format(x_dim, len(X.columns))
                errlog.exit(msg)

        if do_simulation:
            pass
        else:
            Y = read_file(self.outData,errlog)
            if Y.shape[1] != y_dim:
                msg = 'Error importing input data: Number of dimension inconsistent: have {} QoI(s) but {} column(s).' \
                    .format(y_dim, len(Y.columns))
                errlog.exit(msg)

            if X.shape[0] != Y.shape[0]:
                msg = 'Error importing input data: numbers of samples of inputs ({}) and outputs ({}) are inconsistent'.format(len(X.columns), len(Y.columns))
                errlog.exit(msg)

        # writing a data file for PLoM input
        inputXY = os.path.join(work_dir, "templatedir/inputXY.csv")
        X_Y = pd.concat([X,Y], axis=1)
        pd.to_csv(inputXY, sept=',', header=True, index=False)
        self.inputXY = inputXY


    def train_model(self, model_name='SurrogatePLoM'):
        db_path = os.path.join(self.work_dir, 'templatedir')
        self.modelPLoM = PLoM(model_name=model_name, data=self.inputXY, separator=',', col_header=True, db_path=db_path)
        tasks = ['DataNormalization','RunPCA','RunKDE']
        self.modelPLoM.ConfigTasks(task_list=tasks)
        self.modelPLoM.RunAlgorithm()


    def save_model(self):
        pass

def read_file(text_dir, errlog):
    if not os.path.exists(text_dir):
        msg = "Error: file does not exist: " + text_dir
        errlog.exit(msg)
    
    sep_options = [',', '\s+']
    for cur_sep in sep_options:
        try:
            tmp_data = pd.read_csv(text_dir, sep=cur_sep, header=0)
        except:
            tmp_data = None
        if tmp_data:
            break

    if not tmp_data:
        msg = "Error: file is not supported " + text_dir
        errlog.exit(msg)
        X = None
    else:
        X = tmp_data  

    return X
    

class errorLog(object):

    def __init__(self, work_dir):
        self.file = open('{}/dakota.err'.format(work_dir), "w")

    def exit(self, msg):
        print(msg)
        self.file.write(msg)
        self.file.close()
        exit(-1)


def build_surrogate(work_dir, os_type, run_type):
    # t_total = time.process_time()
    filename = 'PLoM_Model'

    f = open(work_dir + '/templatedir/dakota.json')
    try:
        inp = json.load(f)
    except ValueError:
        msg = 'invalid json format - dakota.json'
        errlog.exit(msg)

    f.close()

    if inp['UQ_Method']['uqType'] != 'Train PLoM Model':
        msg = 'UQ type inconsistency : user wanted <' + inp['UQ_Method'][
            'uqType'] + '> but called <Train PLoM Model> program'
        errlog.exit(msg)


    # initializing runPLoM
    model = runPLoM(work_dir, run_type, os_type, inp, errlog)
    # training the model
    model.training()

# the actual execution

# ==========================================================================================

# the actual execution
# this module is extended from the surrogateBuild.py to include using PLoM for surrogate
# modeling while maintaining similar input/output formats compatable with the current workflow

if __name__ == "__main__":
    inputArgs = sys.argv
    work_dir = inputArgs[1].replace(os.sep, '/')
    print(work_dir)

    errlog = errorLog(work_dir)

    run_type = inputArgs[3]
    os_type = inputArgs[2]
    print(run_type)
    print(os_type)
    result_file = "results.out"
    build_surrogate(work_dir, os_type, run_type)    