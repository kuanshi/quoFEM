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

#include "SimCenterUQInputSensitivity.h"
#include <SimCenterUQResultsSensitivity.h>
#include <RandomVariablesContainer.h>


#include <QPushButton>
#include <QScrollArea>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QDebug>
#include <QFileDialog>
#include <QPushButton>
#include <sectiontitle.h>
#include <InputWidgetEDP.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>


#include <QStackedWidget>
#include <MonteCarloInputWidget.h>
#include <LatinHypercubeInputWidget.h>

SimCenterUQInputSensitivity::SimCenterUQInputSensitivity(QWidget *parent)
: UQ_Engine(parent),uqSpecific(0)
{
    mLayout = new QVBoxLayout();

    //
    // create layout for selection box for method type to layout
    //

    QHBoxLayout *methodLayout= new QHBoxLayout;
    QLabel *label1 = new QLabel();
    label1->setText(QString("Method"));
    samplingMethod = new QComboBox();
    samplingMethod->addItem(tr("Monte Carlo"));
    samplingMethod->setMaximumWidth(200);
    samplingMethod->setMinimumWidth(200);


    methodLayout->addWidget(label1);
    methodLayout->addWidget(samplingMethod,2);
    methodLayout->addStretch(4);

    mLayout->addLayout(methodLayout);

    //    mLayout->addStretch(1);

    //
    // qstacked widget to hold all widgets
    //

    theStackedWidget = new QStackedWidget();

    //theLHS = new LatinHypercubeInputWidget();
    //theStackedWidget->addWidget(theLHS);

    theMC = new MonteCarloInputWidget();
    theStackedWidget->addWidget(theMC);

    // set current widget to index 0
    theCurrentMethod = theMC;

    mLayout->addWidget(theStackedWidget);


    //
    // Import paired data
    //

    checkBoxLayout= new QHBoxLayout;
    checkBoxLayout->setMargin(0);
    checkBoxLayout->setAlignment(Qt::AlignTop);

    label2 = new QLabel();
    label2->setText(QString("Resample RVs from correlated dataset"));
    label2->setStyleSheet("font-weight: bold; color: gray");
    importCorrDataCheckBox = new QCheckBox();
    checkBoxLayout->addWidget(importCorrDataCheckBox,0);
    checkBoxLayout->addWidget(label2,1);
    mLayout->addLayout(checkBoxLayout);

    corrDataLayoutWrap= new QWidget;
    QGridLayout *corrDataLayout= new QGridLayout(corrDataLayoutWrap);

    QFrame * lineA = new QFrame;
    lineA->setFrameShape(QFrame::HLine);
    lineA->setFrameShadow(QFrame::Sunken);
    lineA->setMaximumWidth(300);
    corrDataLayout->addWidget(lineA,0,0,1,-1);

    corrDataLayout->setMargin(0);
    QLabel *label3 = new QLabel();
    label3->setText(QString("RV data groups"));
    varList = new QLineEdit();
    varList->setPlaceholderText("e.g. {RV_name1,RV_name2},{RV_name5,RV_name6,RV_name8}");
    varList->setMaximumWidth(420);
    varList->setMinimumWidth(420);
    corrDataLayout->addWidget(label3,1,0);
    corrDataLayout->addWidget(varList,1,1);

    corrDataLayout->setRowStretch(3,1);
    corrDataLayout->setColumnStretch(2,1);

    corrDataLayoutWrap ->setVisible(false);
    mLayout->addWidget(corrDataLayoutWrap);
    mLayout->addStretch(2);
    mLayout->setStretch(3,1);

    this->setLayout(mLayout);

    //connect(samplingMethod, SIGNAL(currentTextChanged(QString)), this, SLOT(onTextChanged(QString)));
    connect(importCorrDataCheckBox,SIGNAL(toggled(bool)),this,SLOT(showDataOptions(bool)));

}

void SimCenterUQInputSensitivity::onMethodChanged(QString text)
{
  if (text=="LHS") {
    //theStackedWidget->setCurrentIndex(0);
    //theCurrentMethod = theLHS;
  }
  else if (text=="Monte Carlo") {
    theStackedWidget->setCurrentIndex(0);
    theCurrentMethod = theMC;  
  }
}

SimCenterUQInputSensitivity::~SimCenterUQInputSensitivity()
{

}

int 
SimCenterUQInputSensitivity::getMaxNumParallelTasks(void){
  return theCurrentMethod->getNumberTasks();
}

void SimCenterUQInputSensitivity::clear(void)
{

}

bool
SimCenterUQInputSensitivity::outputToJSON(QJsonObject &jsonObject)
{
    // testing
    bool result = true;

    QJsonObject uq;
    uq["method"]=samplingMethod->currentText();
    theCurrentMethod->outputToJSON(uq);

    jsonObject["samplingMethodData"]=uq;

    if (importCorrDataCheckBox->isChecked()) {
        jsonObject["RVdataGroup"] = varList->text();
    } else {
        jsonObject["RVdataGroup"] = ""; // empty
    }

    return result;
}


bool
SimCenterUQInputSensitivity::inputFromJSON(QJsonObject &jsonObject)
{
  bool result = false;
  this->clear();

  //
  // get sampleingMethodData, if not present it's an error
  //

  if (jsonObject.contains("samplingMethodData")) {
      QJsonObject uq = jsonObject["samplingMethodData"].toObject();
      if (uq.contains("method")) {

          QString method =uq["method"].toString();
          int index = samplingMethod->findText(method);
          if (index == -1) {
              return false;
          }
          samplingMethod->setCurrentIndex(index);
          result = theCurrentMethod->inputFromJSON(uq);
          if (result == false)
              return result;

      }
    }

   if (jsonObject.contains("RVdataGroup")) {
      varList->setText(jsonObject["RVdataGroup"].toString());
      if ((varList->text()).isEmpty()) {
          importCorrDataCheckBox->setChecked(false);
      } else {
          importCorrDataCheckBox->setChecked(true);
      }
  }

  return result;
}


void SimCenterUQInputSensitivity::showDataOptions(bool tog)
{
    if (tog) {
        label2->setStyleSheet("font-weight: bold; color: black");
        corrDataLayoutWrap->setVisible(true);
    } else {
        label2->setStyleSheet("font-weight: bold; color: gray");
        corrDataLayoutWrap->setVisible(false);
    }
}

int SimCenterUQInputSensitivity::processResults(QString &filenameResults, QString &filenameTab) {
    return 0;
}

UQ_Results *
SimCenterUQInputSensitivity::getResults(void) {
    return new SimCenterUQResultsSensitivity(theRandomVariables);
}

RandomVariablesContainer *
SimCenterUQInputSensitivity::getParameters(void) {
  QString classType("Uncertain");
  theRandomVariables =  new RandomVariablesContainer(classType,tr("SimCenterUQ"));
  return theRandomVariables;
}

QString
SimCenterUQInputSensitivity::getMethodName(void){
  return QString("sensitivity");
}
