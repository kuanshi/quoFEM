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

    """
    runPLoM: class for run a PLoM job
    methods:
        __init__: initialization
        _create_variables: create variable name lists
        _parse_plom_parameters: parse PLoM modeling parameters
        _set_up_parallel: set up paralleling configurations
        _load_variables: load training data
        train_model: model training
        save_model: model saving
    """

    def __init__(self, work_dir, run_type, os_type, job_config, errlog):

        """
        __init__
        input: 
            work_dir: working directory
            run_type: job type
            os_type: operating system type
            job_config: configuration (dtype = dict)
            errlog: error log object
        """
        
        # read inputs
        self.work_dir = work_dir
        self.run_type = run_type
        self.os_type = os_type
        self.errlog = errlog

        # read variable names
        self.x_dim, self.y_dim, self.rv_name, self.g_name = self._create_variables(job_config)

        # read PLoM parameters
        surrogateInfo = job_config["UQ_Method"]["surrogateMethodInfo"]
        if self._parse_plom_parameters(surrogateInfo):
            msg = 'runPLoM.__init__: Error in reading PLoM parameters.'
            self.errlog.exit(msg)

        # parallel setup
        self.do_parallel = surrogateInfo.get("parallelExecution", False)
        if self.do_parallel:
            if self._set_up_parallel():
                msg = 'runPLoM.__init__: Error in setting up parallel.'
                self.errlog.exit(msg)
        else:
            self.pool = 0
            self.cal_interval = 5

        # prepare training data
        if surrogateInfo["method"] == "Import Data File":
            do_sampling = False
            do_simulation = not surrogateInfo["outputData"]
            self.doe_method = "None"  # default
            do_doe = False
            self.inpData = os.path.join(work_dir, "templatedir/inpFile.in")
            if not do_simulation:
                self.outData = os.path.join(work_dir, "templatedir/outFile.in")
        else:
            msg = 'Error reading json: only supporting "Import Data File"'
            errlog.exit(msg)

        # load variables
        if self._load_variables(do_sampling, do_simulation, job_config):
            msg = 'runPLoM.__init__: Error in loading variables.'
            self.errlog.exit(msg)


    def _create_variables(self, job_config):

        """
        create_variables: creating X and Y variables
        input:
            job_config: job configuration dictionary
        output:
            x_dim: dimension of X data
            y_dim: dimension of Y data
            rv_name: random variable name (X data)
            g_name: variable name (Y data)
        """

        # read X and Y variable names
        rv_name = list()
        g_name = list()
        x_dim = 0
        y_dim = 0
        for rv in job_config['randomVariables']:
            rv_name = rv_name + [rv['name']]
            x_dim += 1
        if x_dim == 0:
            msg = 'Error reading json: RV is empty'
            self.errlog.exit(msg)
        for g in job_config['EDP']:
            if g['length']==1: # scalar
                g_name = g_name + [g['name']]
                y_dim += 1
            else: # vector
                for nl in range(g['length']):
                    g_name = g_name + ["{}_{}".format(g['name'],nl+1)]
                    y_dim += 1
        if y_dim == 0:
            msg = 'Error reading json: EDP(QoI) is empty'
            self.errlog.exit(msg)

        # return
        return x_dim, y_dim, rv_name, g_name


    def _parse_plom_parameters(self, surrogateInfo):

        """
        _parse_plom_parameters: parse PLoM parameters from surrogateInfo
        input:
            surrogateInfo: surrogate information dictionary
        output:
            run_flag: 0 - sucess, 1: failure
        """

        run_flag = 0
        try:
            self.n_mc = int(surrogateInfo['newSampleRatio'])
            self.epsilonPCA = surrogateInfo.get("epsilonPCA",1e-6)
            self.smootherKDE = surrogateInfo.get("smootherKDE",25)
            self.randomSeed = surrogateInfo.get("randomSeed",None)
            self.diffMap = surrogateInfo.get("diffusionMaps",True)
            self.logTransform = surrogateInfo.get("logTransform",False)
            self.constraintsFlag = surrogateInfo.get("constraints",False)
            self.kdeTolerance = surrogateInfo.get("kdeTolerance",0.1)
            if self.constraintsFlag:
                self.constraintsFile = os.path.join(work_dir, "templatedir/plomConstraints.py")
            self.numIter = surrogateInfo.get("numIter",50)
            self.tolIter = surrogateInfo.get("tolIter",0.02)
            self.preTrained = surrogateInfo.get("preTrained",False)
            if self.preTrained:
                self.preTrainedModel = os.path.join(work_dir, "templatedir/surrogatePLoM.h5")
        except:
            run_flag = 1

        # return
        return run_flag


    def _set_up_parallel(self):

        """
        _set_up_parallel: set up modules and variables for parallel jobs
        input:
            none
        output:
            run_flag: 0 - sucess, 1 - failure
        """

        run_flag = 0
        try:
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
        except:
            run_flag = 1

        # return
        return run_flag


    def _load_variables(self, do_sampling, do_simulation, job_config):
        
        """
        _load_variables: load variables
        input:
            do_sampling: sampling flag
            do_simulation: simulation flag
            job_config: job configuration dictionary
        output:
            run_flag: 0 - sucess, 1 - failure
        """

        run_flag = 0
        try:
            if do_sampling:
                pass
            else:
                X = read_txt(self.inpData, self.errlog)
                if len(X.columns) != self.x_dim:
                    msg = 'Error importing input data: Number of dimension inconsistent: have {} RV(s) but {} column(s).' \
                        .format(self.x_dim, len(X.columns))
                    errlog.exit(msg)
                if self.logTransform:
                    X = np.log(X)

            if do_simulation:
                pass
            else:
                Y = read_txt(self.outData, self.errlog)
                if Y.shape[1] != self.y_dim:
                    msg = 'Error importing input data: Number of dimension inconsistent: have {} QoI(s) but {} column(s).' \
                        .format(self.y_dim, len(Y.columns))
                    errlog.exit(msg)
                if self.logTransform:
                    Y = np.log(Y)

                if X.shape[0] != Y.shape[0]:
                    msg = 'Error importing input data: numbers of samples of inputs ({}) and outputs ({}) are inconsistent'.format(len(X.columns), len(Y.columns))
                    errlog.exit(msg)

                n_samp = Y.shape[0]
                # writing a data file for PLoM input
                self.X = X.to_numpy()
                self.Y = Y.to_numpy()
                inputXY = os.path.join(work_dir, "templatedir/inputXY.csv")
                X_Y = pd.concat([X,Y], axis=1)
                X_Y.to_csv(inputXY, sep=',', header=True, index=False)
                self.inputXY = inputXY
                self.n_samp = n_samp

                self.do_sampling = do_sampling
                self.do_simulation = do_simulation
                self.rvName = []
                self.rvDist = []
                self.rvVal = []
                for nx in range(self.x_dim):
                    rvInfo = job_config["randomVariables"][nx]
                    self.rvName = self.rvName + [rvInfo["name"]]
                    self.rvDist = self.rvDist + [rvInfo["distribution"]]
                    if do_sampling:
                        self.rvVal = self.rvVal + [(rvInfo["upperbound"] + rvInfo["lowerbound"]) / 2]
                    else:
                        self.rvVal = self.rvVal + [np.mean(self.X[:, nx])]
        except:
            run_flag = 1

        # return
        return run_flag


    def train_model(self, model_name='SurrogatePLoM'):
        db_path = os.path.join(self.work_dir, 'templatedir')
        if not self.preTrained:
            self.modelPLoM = PLoM(model_name=model_name, data=self.inputXY, separator=',', col_header=True, db_path=db_path, 
                tol_pca = self.epsilonPCA, epsilon_kde = self.smootherKDE, runDiffMaps = self.diffMap, plot_tag = True)
        else:
            self.modelPLoM = PLoM(model_name=model_name, data=self.preTrainedModel, db_path=db_path, 
                tol_pca = self.epsilonPCA, epsilon_kde = self.smootherKDE, runDiffMaps = self.diffMap)
        if self.constraintsFlag:
            self.modelPLoM.add_constraints(self.constraintsFile)
        if self.n_mc > 0:
            tasks = ['DataNormalization','RunPCA','RunKDE','ISDEGeneration']
        else:
            tasks = ['DataNormalization','RunPCA','RunKDE']
        self.modelPLoM.ConfigTasks(task_list=tasks)
        self.modelPLoM.RunAlgorithm(n_mc=self.n_mc, tol = self.tolIter, max_iter = self.numIter, seed_num=self.randomSeed, tolKDE=self.kdeTolerance)
        if self.n_mc > 0:
            self.modelPLoM.export_results(data_list=['/X0','/X_new'])
        else:
            self.modelPLoM.export_results(data_list=['/X0'])
        self.pcaEigen = self.modelPLoM.mu
        self.pcaError = self.modelPLoM.errPCA
        self.pcaComp = self.modelPLoM.nu
        self.kdeEigen = self.modelPLoM.eigenKDE
        self.kdeComp = self.modelPLoM.m
        self.Errors = []
        if self.constraintsFlag:
            self.Errors = self.modelPLoM.errors


    def save_model(self):

        # copy the h5 model file to the main work dir
        shutil.copy2(os.path.join(self.work_dir,'templatedir','SurrogatePLoM','SurrogatePLoM.h5'),self.work_dir)
        if self.n_mc > 0:
            shutil.copy2(os.path.join(self.work_dir,'templatedir','SurrogatePLoM','DataOut','X_new.csv'),self.work_dir)

        header_string_x = ' ' + ' '.join([str(elem) for elem in self.rv_name]) + ' '
        header_string_y = ' ' + ' '.join([str(elem) for elem in self.g_name])
        header_string = header_string_x + header_string_y

        #xy_data = np.concatenate((np.asmatrix(np.arange(1, self.n_samp + 1)).T, self.X, self.Y), axis=1)
        #np.savetxt(self.work_dir + '/dakotaTab.out', xy_data, header=header_string, fmt='%1.4e', comments='%')
        np.savetxt(self.work_dir + '/inputTab.out', self.X, header=header_string_x, fmt='%1.4e', comments='%')
        np.savetxt(self.work_dir + '/outputTab.out', self.Y, header=header_string_y, fmt='%1.4e', comments='%')

        results = {}

        results["valSamp"] = self.n_samp
        results["xdim"] = self.x_dim
        results["ydim"] = self.y_dim
        results["xlabels"] = self.rv_name
        results["ylabels"] = self.g_name
        results["yExact"] = {}
        results["xPredict"] = {}
        results["yPredict"] = {}
        results["valNRMSE"] = {}
        results["valR2"] = {}
        results["valCorrCoeff"] = {}
        for ny in range(self.y_dim):
            results["yExact"][self.g_name[ny]] = self.Y[:, ny].tolist()

        results["inpData"] = self.inpData
        if not self.do_simulation:
            results["outData"] = self.outData

        results["logTransform"] = self.logTransform

        rv_list = []
        for nx in range(self.x_dim):
            rvs = {}
            rvs["name"] = self.rvName[nx]
            rvs["distribution"] = self.rvDist[nx]
            rvs["value"] = self.rvVal[nx]
            rv_list = rv_list + [rvs]
        results["randomVariables"] = rv_list
        results["dirPLoM"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),'PLoM')

        results["pcaEigen"] = self.pcaEigen.tolist()
        results["pcaError"] = self.pcaError
        results["pcaComp"] = self.pcaComp
        results["kdeEigen"] = self.kdeEigen.tolist()
        results["kdeComp"] = self.kdeComp
        results["Errors"] = self.Errors
        
        if self.n_mc > 0:
            Xnew = pd.read_csv(self.work_dir + '/X_new.csv', header=0, index_col=0)
            if self.logTransform:
                Xnew = np.exp(Xnew)
            for nx in range(self.x_dim):
                results["xPredict"][self.rv_name[nx]] = Xnew.iloc[:, nx].tolist()

            for ny in range(self.y_dim):
                results["yPredict"][self.g_name[ny]] = Xnew.iloc[:, self.x_dim+ny].tolist()

        xy_data = np.concatenate((np.asmatrix(np.arange(1, self.X.shape[0] + 1)).T, self.X, self.Y), axis=1)
        np.savetxt(self.work_dir + '/dakotaTab.out', xy_data, header=header_string, fmt='%1.4e', comments='%')

            #if not self.do_logtransform:
            #results["yPredict_CI_lb"][self.g_name[ny]] = norm.ppf(0.25, loc = results["yPredict"][self.g_name[ny]] , scale = np.sqrt(self.Y_loo_var[:, ny])).tolist()
            #results["yPredict_CI_ub"][self.g_name[ny]] = norm.ppf(0.75, loc = results["yPredict"][self.g_name[ny]] , scale = np.sqrt(self.Y_loo_var[:, ny])).tolist()
            #else:
            #    mu = np.log(self.Y_loo[:, ny] )
            #    sig = np.sqrt(np.log(self.Y_loo_var[:, ny]/pow(self.Y_loo[:, ny] ,2)+1))
            #    results["yPredict_CI_lb"][self.g_name[ny]] =  lognorm.ppf(0.25, s = sig, scale = np.exp(mu)).tolist()
            #    results["yPredict_CI_ub"][self.g_name[ny]] =  lognorm.ppf(0.75, s = sig, scale = np.exp(mu)).tolist()

        with open('dakota.out', 'w') as fp:
            json.dump(results, fp, indent=2)

        print("Results Saved")


def read_txt(text_dir, errlog):

    if not os.path.exists(text_dir):
        msg = "Error: file does not exist: " + text_dir
        errlog.exit(msg)

    with open(text_dir) as f:
        # Iterate through the file until the table starts
        header_count = 0
        for line in f:
            if line.startswith('%'):
                header_count = header_count + 1
                print(line)
        try:
            with open(text_dir) as f:
                X = np.loadtxt(f, skiprows=header_count)
        except ValueError:
            with open(text_dir) as f:
                try:
                    X = np.genfromtxt(f, skip_header=header_count, delimiter=',')
                    # if there are extra delimiter, remove nan
                    if np.isnan(X[-1, -1]):
                        X = np.delete(X, -1, 1)
                except ValueError:
                    msg = "Error: file format is not supported " + text_dir
                    errlog.exit(msg)

    if X.ndim == 1:
        X = np.array([X]).transpose()

    df_X = pd.DataFrame(data=X, columns=["V"+str(x) for x in range(X.shape[1])])

    return df_X
    

class errorLog(object):

    def __init__(self, work_dir):
        self.file = open('{}/dakota.err'.format(work_dir), "w")

    def exit(self, msg):
        print(msg)
        self.file.write(msg)
        self.file.close()
        exit(-1)


def build_surrogate(work_dir, os_type, run_type):
    
    """
    build_surrogate: built surrogate model
    input:
        work_dir: working directory
        run_type: job type
        os_type: operating system type
    """

    # t_total = time.process_time()
    # default filename
    filename = 'PLoM_Model'
    # read the configuration file
    f = open(work_dir + '/templatedir/dakota.json')
    try:
        job_config = json.load(f)
    except ValueError:
        msg = 'invalid json format - dakota.json'
        errlog.exit(msg)
    f.close()

    # check the uq type
    if job_config['UQ_Method']['uqType'] != 'PLoM Model':
        msg = 'UQ type inconsistency : user wanted <' + job_config['UQ_Method']['uqType'] + \
            '> but called <PLoM Model> program'
        errlog.exit(msg)

    # initializing runPLoM
    model = runPLoM(work_dir, run_type, os_type, job_config, errlog)
    # training the model
    model.train_model()
    # save the model
    model.save_model()


if __name__ == "__main__":

    """
    shell command: PYTHON runPLoM.py work_dir run_type os_type
    work_dir: working directory
    run_type: job type
    os_type: operating system type
    """

    # collect arguments
    inputArgs = sys.argv
    # working diretory
    work_dir = inputArgs[1].replace(os.sep, '/')
    # print the work_dir
    errlog = errorLog(work_dir)
    # job type
    run_type = inputArgs[3]
    # operating system type
    os_type = inputArgs[2]
    # default output file: results.out
    result_file = "results.out"
    # start build the surrogate
    build_surrogate(work_dir, os_type, run_type)    