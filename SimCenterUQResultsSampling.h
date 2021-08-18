#ifndef SimCenterUQ_RESULTS_SAMPLING_H
#define SimCenterUQ_RESULTS_SAMPLING_H

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

#include <UQ_Results.h>
#include <QtCharts/QChart>
#include <QMessageBox>
#include <QPushButton>
#include <ResultsDataChart.h>


using namespace QtCharts;

class QTextEdit;
class QTabWidget;
class MyTableWidget;
class MainWindow;
class RandomVariablesContainer;

//class QChart;

class SimCenterUQResultsSampling : public UQ_Results
{
    Q_OBJECT
public:
  explicit SimCenterUQResultsSampling(RandomVariablesContainer *, QWidget *parent = 0);
    ~SimCenterUQResultsSampling();

    bool outputToJSON(QJsonObject &rvObject);
    bool inputFromJSON(QJsonObject &rvObject);

    int processResults(QString &filenameResults, QString &filenameTab);
    QWidget *createResultEDPWidget(QString &name, QVector<double> statistics);

signals:

public slots:
   void clear(void);
//   void onSpreadsheetCellClicked(int, int);
//   void onSaveSpreadsheetClicked();
//   void onSaveSpreadsheetSeparatelyClicked();
   // modified by padhye 08/25/2018

private:
   RandomVariablesContainer *theRVs;
   QTabWidget *tabWidget;

   //MyTableWidget *spreadsheet;  // MyTableWidget inherits the QTableWidget
   //QChart *chart;
   //QPushButton* save_spreadheet; // save the data from spreadsheet
   //QLabel *label;
   //QLabel *best_fit_instructions;


   QVector<QString>theNames;
   QVector<double>theMeans;
   QVector<double>theStdDevs;
   QVector<double>theKurtosis;
   QVector<double>theSkewness;

   ResultsDataChart * theDataTable;
};

#endif // SimCenterUQ_RESULTS_SAMPLING_H
