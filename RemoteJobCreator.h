#ifndef REMOTEJOBCREATOR_H
#define REMOTEJOBCREATOR_H

/* *****************************************************************************
Copyright (c) 2016-2017, The Regents of the University of California (Regents).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

*************************************************************************** */


// Written: fmckenna

// Purpose: a widget for submitting uqFEM jobs to HPC resource (specifically DesignSafe at moment)
//  - the widget aasks for additional info needed and provide a submit button to submit the jb when clicked.

#include <QWidget>
#include <AgaveCurl.h>

class QLineEdit;

class JobManager;
class MainWindow;
class QPushButton;

class RemoteJobCreator : public QWidget
{
    Q_OBJECT
public:
    explicit RemoteJobCreator(AgaveCurl *, QWidget *parent = nullptr);
    void setInputDirectory(const QString & directoryName);
    void setMaxNumParallelProcesses(int);

signals:
   void getHomeDirCall(void);
   void uploadDirCall(const QString &local, const QString &remote);
   void startJobCall(QJsonObject theJob);
   void successfullJobStart(void);
   void errorMessage(QString);
   void statusMessage(QString);
   void fatalMessage(QString);

public slots:
    void attemptLoginReturn(bool);

    void pushButtonClicked(void);
    void uploadDirReturn(bool);
    void getHomeDirReturned(QString);
    void startJobReturn(QString);


private:
    void submitJob(void);

    QLineEdit *nameLineEdit;
    QLineEdit *numCPU_LineEdit;
    QLineEdit *numProcessorsLineEdit;
    QLineEdit *runtimeLineEdit;
    QLineEdit *appLineEdit;

    AgaveCurl   *theInterface;
 //   JobManager *theManager;

    QString directoryName;
    QPushButton *pushButton;

    QString remoteHomeDirPath;
    QJsonObject theJob;
    int maxParallel;
};

#endif // REMOTEJOBCREATOR_H
