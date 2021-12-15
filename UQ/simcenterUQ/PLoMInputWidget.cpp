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
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
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

// Written: fmckenna, kuanshi

#include <PLoMInputWidget.h>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QLabel>
#include <QValidator>
#include <QJsonObject>
#include <QPushButton>
#include <QFileDialog>
#include <QCheckBox>
#include <QComboBox>
#include <iostream>
#include <fstream>
#include <regex>
#include <iterator>
#include <string>
#include <sstream>
#include <InputWidgetParameters.h>
#include <InputWidgetEDP.h>
#include <InputWidgetFEM.h>
#include <QButtonGroup>
#include <QRadioButton>
#include <QStackedWidget>

PLoMInputWidget::PLoMInputWidget(InputWidgetParameters *param,InputWidgetFEM *femwidget,InputWidgetEDP *edpwidget, QWidget *parent)
: UQ_MethodInputWidget(parent), theParameters(param), theEdpWidget(edpwidget), theFemWidget(femwidget)
{
    auto layout = new QGridLayout();
    int wid = 0; // widget id

    //First we need to add type radio buttons
    m_typeButtonsGroup = new QButtonGroup(this);
    QRadioButton* rawDataRadioButton = new QRadioButton(tr("Raw Data"));
    QRadioButton* preTrainRadioButton = new QRadioButton(tr("Pre-trained Model"));
    m_typeButtonsGroup->addButton(rawDataRadioButton, 0);
    m_typeButtonsGroup->addButton(preTrainRadioButton, 1);
    QWidget* typeGroupBox = new QWidget(this);
    typeGroupBox->setContentsMargins(0,0,0,0);
    typeGroupBox->setStyleSheet("QGroupBox { font-weight: normal;}");
    QHBoxLayout* typeLayout = new QHBoxLayout(typeGroupBox);
    typeGroupBox->setLayout(typeLayout);
    typeLayout->addWidget(rawDataRadioButton);
    typeLayout->addWidget(preTrainRadioButton);
    layout->addWidget(typeGroupBox,wid++,0);

    rawDataGroup = new QWidget(this);
    QGridLayout* rawDataLayout = new QGridLayout(rawDataGroup);
    rawDataGroup->setLayout(rawDataLayout);
    preTrainGroup = new QWidget(this);
    QGridLayout* preTrainLayout = new QGridLayout(preTrainGroup);
    preTrainGroup->setLayout(preTrainLayout);


    //
    // Create Input LineEdit
    //

    inpFileDir = new QLineEdit();
    QPushButton *chooseInpFile = new QPushButton("Choose");
    connect(chooseInpFile, &QPushButton::clicked, this, [=](){
        inpFileDir->setText(QFileDialog::getOpenFileName(this,tr("Open File"),"", "All files (*.*)"));
        this->parseInputDataForRV(inpFileDir->text());
    });
    inpFileDir->setMinimumWidth(200);
    inpFileDir->setReadOnly(true);
    //layout->addWidget(new QLabel("Training Data File: Input Variable"),wid,0);
    //layout->addWidget(inpFileDir,wid,1,1,3);
    //layout->addWidget(chooseInpFile,wid++,4);
    rawDataLayout->addWidget(new QLabel("Training Data File: Input"),0,0);
    rawDataLayout->addWidget(inpFileDir,0,1,1,3);
    rawDataLayout->addWidget(chooseInpFile,0,4);

    //
    // Create Output LineEdit
    //
    outFileDir = new QLineEdit();
    chooseOutFile = new QPushButton("Choose");
    connect(chooseOutFile, &QPushButton::clicked, this, [=](){
        outFileDir->setText(QFileDialog::getOpenFileName(this,tr("Open File"),"", "All files (*.*)"));
        this->parseOutputDataForQoI(outFileDir->text());
    });
    outFileDir->setMinimumWidth(200);
    outFileDir->setReadOnly(true);
    //layout->addWidget(new QLabel("Training Data File: Output Response"),wid,0,Qt::AlignTop);
    //layout->addWidget(outFileDir,wid,1,1,3,Qt::AlignTop);
    //layout->addWidget(chooseOutFile,wid,4,Qt::AlignTop);
    rawDataLayout->addWidget(new QLabel("Training Data File: Output"),1,0,Qt::AlignTop);
    rawDataLayout->addWidget(outFileDir,1,1,1,3,Qt::AlignTop);
    rawDataLayout->addWidget(chooseOutFile,1,4,Qt::AlignTop);

    errMSG=new QLabel("Unrecognized file format");
    errMSG->setStyleSheet({"color: red"});
    //layout->addWidget(errMSG,wid++,1,Qt::AlignLeft);
    rawDataLayout->addWidget(errMSG,2,1,Qt::AlignLeft);
    errMSG->hide();

    inpFileDir2 = new QLineEdit();
    QPushButton *chooseInpFile2 = new QPushButton("Choose");
    connect(chooseInpFile2, &QPushButton::clicked, this, [=](){
        inpFileDir2->setText(QFileDialog::getOpenFileName(this,tr("Open File"),"", "All files (*.*)"));
        this->parseInputDataForRV(inpFileDir2->text());
    });
    inpFileDir2->setMinimumWidth(200);
    inpFileDir2->setReadOnly(true);
    preTrainLayout->addWidget(new QLabel("Training Data File: Pretrained Model"),0,0);
    preTrainLayout->addWidget(inpFileDir2,0,1,1,3);
    preTrainLayout->addWidget(chooseInpFile2,0,4);

    //We will add stacked widget to switch between grid and single location
    m_stackedWidgets = new QStackedWidget(this);
    m_stackedWidgets->addWidget(rawDataGroup);
    m_stackedWidgets->addWidget(preTrainGroup);

    m_typeButtonsGroup->button(0)->setChecked(true);
    m_stackedWidgets->setCurrentIndex(0);
    connect(m_typeButtonsGroup, QOverload<int>::of(&QButtonGroup::buttonReleased), [this](int id)
    {
        if(id == 0) {
            m_typeButtonsGroup->button(0)->setChecked(true);
            m_stackedWidgets->setCurrentIndex(0);
            preTrained = false;
        }
        else if (id == 1) {
            m_typeButtonsGroup->button(1)->setChecked(true);
            m_stackedWidgets->setCurrentIndex(1);
            preTrained = true;
        }
    });
    layout->addWidget(m_stackedWidgets,wid++,0);

    //
    // create layout label and entry for # of new samples
    //

    ratioNewSamples = new QLineEdit();
    ratioNewSamples->setText(tr("5"));
    ratioNewSamples->setValidator(new QIntValidator);
    ratioNewSamples->setToolTip("The ratio between the number of new realizations and the size of original sample. \nIf \"0\" is given, the PLoM model is trained without new predictions");
    ratioNewSamples->setMaximumWidth(150);

    QLabel *newSNR = new QLabel("New Sample Number Ratio");
    layout->addWidget(newSNR, wid, 0);
    layout->addWidget(ratioNewSamples, wid++, 1);


    //
    // Create Advanced options
    //

    theAdvancedCheckBox = new QCheckBox();
    theAdvancedTitle=new QLabel("\n     Advanced Options");
    theAdvancedTitle->setStyleSheet("font-weight: bold; color: gray");
    layout->addWidget(theAdvancedTitle, wid, 0,1,3,Qt::AlignBottom);
    layout->addWidget(theAdvancedCheckBox, wid++, 0,Qt::AlignBottom);

    lineA = new QFrame;
    lineA->setFrameShape(QFrame::HLine);
    lineA->setFrameShadow(QFrame::Sunken);
    lineA->setMaximumWidth(420);
    layout->addWidget(lineA, wid++, 0, 1, 3);
    lineA->setVisible(false);

    //
    // Use Log transform
    //

    theLogtLabel=new QLabel("Log-space Transform");
    theLogtLabel2=new QLabel("     (check this box only when all data are always positive)");

    theLogtCheckBox = new QCheckBox();
    layout->addWidget(theLogtLabel, wid, 0);
    layout->addWidget(theLogtLabel2, wid, 1,1,-1,Qt::AlignLeft);
    layout->addWidget(theLogtCheckBox, wid++, 1);
    theLogtLabel->setVisible(false);
    theLogtLabel2->setVisible(false);
    theLogtCheckBox->setVisible(false);

    //
    theDMLabel=new QLabel("Diffusion Maps");
    theDMCheckBox = new QCheckBox();
    layout->addWidget(theDMLabel, wid, 0);
    layout->addWidget(theDMCheckBox, wid++, 1);
    theDMLabel->setVisible(false);
    theDMCheckBox->setVisible(false);
    theDMCheckBox->setChecked(true);

    //
    epsilonPCA = new QLineEdit();
    epsilonPCA->setText(tr("0.0001"));
    epsilonPCA->setValidator(new QDoubleValidator);
    epsilonPCA->setToolTip("PCA Tolerance");
    epsilonPCA->setMaximumWidth(150);
    newEpsilonPCA = new QLabel("PCA Tolerance");
    layout->addWidget(newEpsilonPCA, wid, 0);
    layout->addWidget(epsilonPCA, wid++, 1);
    epsilonPCA->setVisible(false);
    newEpsilonPCA->setVisible(false);

    //
    smootherKDE = new QLineEdit();
    smootherKDE->setText(tr("25"));
    smootherKDE->setValidator(new QDoubleValidator);
    smootherKDE->setToolTip("KDE Smooth Factor");
    smootherKDE->setMaximumWidth(150);
    newSmootherKDE = new QLabel("KDE Smooth Factor");
    layout->addWidget(newSmootherKDE, wid, 0);
    layout->addWidget(smootherKDE, wid++, 1);
    smootherKDE->setVisible(false);
    newSmootherKDE->setVisible(false);

    //
    tolKDE = new QLineEdit();
    tolKDE->setText(tr("0.1"));
    tolKDE->setValidator(new QDoubleValidator);
    tolKDE->setToolTip("KDE Tolerance: ratio between the cut-off eigenvalue and the first eigenvalue.");
    tolKDE->setMaximumWidth(150);
    newTolKDE = new QLabel("KDE Tolerance");
    layout->addWidget(newTolKDE, wid, 0);
    layout->addWidget(tolKDE, wid++, 1);
    tolKDE->setVisible(false);
    newTolKDE->setVisible(false);

    //
    randomSeed = new QLineEdit();
    randomSeed->setText(tr("10"));
    randomSeed->setValidator(new QIntValidator);
    randomSeed->setToolTip("Random Seed Number");
    randomSeed->setMaximumWidth(150);
    newRandomSeed = new QLabel("Random Seed");
    layout->addWidget(newRandomSeed, wid, 0);
    layout->addWidget(randomSeed, wid++, 1);
    randomSeed->setVisible(false);
    newRandomSeed->setVisible(false);

    // constraints
    QHBoxLayout *theConstraintsLayout = new QHBoxLayout();
    theConstraintsButton = new QCheckBox();
    theConstraintsLabel2 = new QLabel();
    theConstraintsLabel2->setText("Add constratins");
    theConstraintsLayout->addWidget(theConstraintsLabel2,0);
    theConstraintsLayout->addWidget(theConstraintsLabel2,1);
    constraintsPath = new QLineEdit();
    chooseConstraints = new QPushButton(tr("Choose"));
    connect(chooseConstraints, &QPushButton::clicked, this, [=](){
        constraintsPath->setText(QFileDialog::getOpenFileName(this,tr("Open File"),"", "All files (*.*)"));
    });
    constraintsPath->setMinimumWidth(200);
    constraintsPath->setReadOnly(true);
    theConstraintsLabel1 = new QLabel();
    theConstraintsLabel1->setText("Constraints file (.py)");
    layout->addWidget(theConstraintsLabel1,wid,0,Qt::AlignTop);
    layout->addWidget(constraintsPath,wid,1,1,3,Qt::AlignTop);
    layout->addWidget(chooseConstraints,wid,4,Qt::AlignTop);
    layout->addWidget(theConstraintsButton,wid,5,Qt::AlignTop);
    layout->addWidget(theConstraintsLabel2,wid++,6,Qt::AlignTop);
    constraintsPath->setVisible(false);
    theConstraintsLabel1->setVisible(false);
    theConstraintsLabel2->setVisible(false);
    chooseConstraints->setVisible(false);
    theConstraintsButton->setVisible(false);
    constraintsPath->setDisabled(1);
    chooseConstraints->setDisabled(1);
    chooseConstraints->setStyleSheet("background-color: lightgrey;border-color:grey");
    constraintsPath->setStyleSheet("background-color: lightgrey;border-color:grey");
    connect(theConstraintsButton,SIGNAL(toggled(bool)),this,SLOT(setConstraints(bool)));

    // iterations when applying constraints
    numIter = new QLineEdit();
    numIter->setText(tr("50"));
    numIter->setValidator(new QIntValidator);
    numIter->setToolTip("Iteration Number");
    numIter->setMaximumWidth(150);
    numIterLabel = new QLabel("Iteration Number");
    layout->addWidget(numIterLabel, wid, 0);
    layout->addWidget(numIter, wid++, 1);
    numIter->setVisible(false);
    numIterLabel->setVisible(false);
    numIter->setDisabled(1);
    numIter->setStyleSheet("background-color: lightgrey;border-color:grey");

    tolIter = new QLineEdit();
    tolIter->setText(tr("0.02"));
    tolIter->setValidator(new QDoubleValidator);
    tolIter->setToolTip("Iteration Tolerance");
    tolIter->setMaximumWidth(150);
    tolIterLabel = new QLabel("Iteration Tolerance");
    layout->addWidget(tolIterLabel, wid, 0);
    layout->addWidget(tolIter, wid++, 1);
    tolIter->setVisible(false);
    tolIterLabel->setVisible(false);
    tolIter->setDisabled(1);
    tolIter->setStyleSheet("background-color: lightgrey;border-color:grey");

    //
    // Finish
    //

    layout->setRowStretch(wid, 1);
    layout->setColumnStretch(6, 1);
    this->setLayout(layout);

    outFileDir->setDisabled(0);
    chooseOutFile->setDisabled(0);
    //chooseOutFile->setStyleSheet("background-color: lightgrey;border-color:grey");

    connect(theAdvancedCheckBox,SIGNAL(toggled(bool)),this,SLOT(doAdvancedSetup(bool)));


}


PLoMInputWidget::~PLoMInputWidget()
{

}


// SLOT function
void PLoMInputWidget::doAdvancedSetup(bool tog)
{
    if (tog) {
        theAdvancedTitle->setStyleSheet("font-weight: bold; color: black");
    } else {
        theAdvancedTitle->setStyleSheet("font-weight: bold; color: gray");
        theLogtCheckBox->setChecked(false);
    }

    lineA->setVisible(tog);
    theLogtCheckBox->setVisible(tog);
    theLogtLabel->setVisible(tog);
    theLogtLabel2->setVisible(tog);
    theDMCheckBox->setVisible(tog);
    theDMLabel->setVisible(tog);
    epsilonPCA->setVisible(tog);
    newEpsilonPCA->setVisible(tog);
    smootherKDE->setVisible(tog);
    newSmootherKDE->setVisible(tog);
    tolKDE->setVisible(tog);
    newTolKDE->setVisible(tog);
    randomSeed->setVisible(tog);
    newRandomSeed->setVisible(tog);
    constraintsPath->setVisible(tog);
    theConstraintsLabel1->setVisible(tog);
    theConstraintsLabel2->setVisible(tog);
    chooseConstraints->setVisible(tog);
    theConstraintsButton->setVisible(tog);
    numIter->setVisible(tog);
    numIterLabel->setVisible(tog);
    tolIter->setVisible(tog);
    tolIterLabel->setVisible(tog);

}


void PLoMInputWidget::setOutputDir(bool tog)
{
    if (tog) {
        outFileDir->setDisabled(0);
        chooseOutFile->setDisabled(0);
        chooseOutFile->setStyleSheet("color: white");
        theFemWidget->setFEMforGP("GPdata");
        parseInputDataForRV(inpFileDir->text());
        parseOutputDataForQoI(outFileDir->text());
    } else {
        outFileDir->setDisabled(1);
        chooseOutFile->setDisabled(1);
        chooseOutFile->setStyleSheet("background-color: lightgrey;border-color:grey");
        theEdpWidget->setGPQoINames(QStringList("") );
        outFileDir->setText(QString("") );
        theFemWidget->setFEMforGP("GPmodel");
        theFemWidget->femProgramChanged("OpenSees");
        theEdpWidget->setGPQoINames(QStringList({}) );// remove GP RVs
        theParameters->setGPVarNamesAndValues(QStringList({}));// remove GP RVs
    }
}

void PLoMInputWidget::setConstraints(bool tog)
{
    if (tog) {
        constraintsPath->setDisabled(0);
        chooseConstraints->setDisabled(0);
        chooseConstraints->setStyleSheet("background-color: white");
        constraintsPath->setStyleSheet("color: white");
        numIter->setStyleSheet("background-color: white");
        tolIter->setStyleSheet("background-color: white");
        numIter->setDisabled(0);
        tolIter->setDisabled(0);
    } else {
        constraintsPath->setDisabled(1);
        chooseConstraints->setDisabled(1);
        chooseConstraints->setStyleSheet("background-color: lightgrey;border-color:grey");
        constraintsPath->setStyleSheet("background-color: lightgrey;border-color:grey");
        numIter->setStyleSheet("background-color: lightgrey;border-color:grey");
        tolIter->setStyleSheet("background-color: lightgrey;border-color:grey");
        numIter->setDisabled(1);
        tolIter->setDisabled(1);
    }
}

bool
PLoMInputWidget::outputToJSON(QJsonObject &jsonObj){

    bool result = true;

    if (m_typeButtonsGroup->button(0)->isChecked()) {
        jsonObj["preTrained"] = false;
        jsonObj["inpFile"]=inpFileDir->text();
        jsonObj["outFile"]=outFileDir->text();
    } else {
        jsonObj["preTrained"] = true;
        jsonObj["inpFile2"] = inpFileDir2->text();
    }


    jsonObj["outputData"]=true;

    jsonObj["newSampleRatio"]=ratioNewSamples->text().toInt();

    jsonObj["advancedOpt"]=theAdvancedCheckBox->isChecked();
    if (theAdvancedCheckBox->isChecked())
    {
        jsonObj["logTransform"]=theLogtCheckBox->isChecked();
        jsonObj["diffusionMaps"] = theDMCheckBox->isChecked();
        jsonObj["randomSeed"] = randomSeed->text().toInt();
        jsonObj["epsilonPCA"] = epsilonPCA->text().toDouble();
        jsonObj["smootherKDE"] = smootherKDE->text().toDouble();
        jsonObj["kdeTolerance"] = tolKDE->text().toDouble();
        jsonObj["constraints"]= theConstraintsButton->isChecked();
        if (theConstraintsButton->isChecked()) {
            jsonObj["constraintsFile"] = constraintsPath->text();
            jsonObj["numIter"] = numIter->text().toInt();
            jsonObj["tolIter"] = tolIter->text().toDouble();
        }
    }
    jsonObj["parallelExecution"]=false;

    return result;    
}


int PLoMInputWidget::parseInputDataForRV(QString name1){

    double numberOfColumns=countColumn(name1);

    QStringList varNamesAndValues;
    for (int i=0;i<numberOfColumns;i++) {
        varNamesAndValues.append(QString("RV_column%1").arg(i+1));
        varNamesAndValues.append("nan");
    }
    theParameters->setGPVarNamesAndValues(varNamesAndValues);
    numSamples=0;
    return 0;
}

int PLoMInputWidget::parseOutputDataForQoI(QString name1){
    // get number of columns
    double numberOfColumns=countColumn(name1);
    QStringList qoiNames;
    for (int i=0;i<numberOfColumns;i++) {
        qoiNames.append(QString("QoI_column%1").arg(i+1));
    }
    theEdpWidget->setGPQoINames(qoiNames);
    return 0;
}

int PLoMInputWidget::countColumn(QString name1){
    // get number of columns
    std::ifstream inFile(name1.toStdString());
    // read lines of input searching for pset using regular expression
    std::string line;
    errMSG->hide();

    int numberOfColumns_pre = -100;
    while (getline(inFile, line)) {
        int  numberOfColumns=1;
        bool previousWasSpace=false;
        //for(int i=0; i<line.size(); i++){
        for(size_t i=0; i<line.size(); i++){
            if(line[i] == '%' || line[i] == '#'){ // ignore header
                numberOfColumns = numberOfColumns_pre;
                break;
            }
            if(line[i] == ' ' || line[i] == '\t' || line[i] == ','){
                if(!previousWasSpace)
                    numberOfColumns++;
                previousWasSpace = true;
            } else {
                previousWasSpace = false;
            }
        }
        if(previousWasSpace)// when there is a blank space at the end of each row
            numberOfColumns--;

        if (numberOfColumns_pre==-100)  // to pass header
        {
            numberOfColumns_pre=numberOfColumns;
            continue;
        }
        if (numberOfColumns != numberOfColumns_pre)// Send an error
        {
            errMSG->show();
            numberOfColumns_pre=0;
            break;
        }
    }
    // close file
    inFile.close();
    return numberOfColumns_pre;
}

bool
PLoMInputWidget::inputFromJSON(QJsonObject &jsonObject){

    bool result = false;
    preTrained = false;
    if (jsonObject.contains("preTrained")) {
        preTrained = jsonObject["preTrained"].toBool();
        result = true;
    } else {
        return false;
    }
    if (preTrained) {
        if (jsonObject.contains("inpFile2")) {
            QString fileDir=jsonObject["inpFile2"].toString();
            inpFileDir2->setText(fileDir);
            result = true;
        } else {
            return false;
        }

    } else {
        if (jsonObject.contains("inpFile")) {
            QString fileDir=jsonObject["inpFile"].toString();
            inpFileDir->setText(fileDir);
            result = true;
        } else {
            return false;
        }

        if (jsonObject.contains("outputData")) {
          if (jsonObject["outputData"].toBool()) {
              QString fileDir=jsonObject["outFile"].toString();
              outFileDir->setText(fileDir);
              theFemWidget->setFEMforGP("GPdata");
          }
          result = true;
        } else {
          return false;
        }
    }


    if (jsonObject.contains("newSampleRatio")) {
        int samples=jsonObject["newSampleRatio"].toInt();
        ratioNewSamples->setText(QString::number(samples));
    } else {
        result = false;
    }

  if (jsonObject.contains("advancedOpt")) {
      theAdvancedCheckBox->setChecked(jsonObject["advancedOpt"].toBool());
      if (jsonObject["advancedOpt"].toBool()) {
        theAdvancedCheckBox->setChecked(true);
        theLogtCheckBox->setChecked(jsonObject["logTransform"].toBool());
        theDMCheckBox->setChecked(jsonObject["diffusionMaps"].toBool());
        randomSeed->setText(QString::number(jsonObject["randomSeed"].toInt()));
        smootherKDE->setText(QString::number(jsonObject["smootherKDE"].toDouble()));
        tolKDE->setText(QString::number(jsonObject["kdeTolerance"].toDouble()));
        epsilonPCA->setText(QString::number(jsonObject["epsilonPCA"].toDouble()));
        theConstraintsButton->setChecked(jsonObject["constraints"].toBool());
        if (jsonObject["constraints"].toBool()) {
            constraintsPath->setText(jsonObject["constraintsFile"].toString());
            numIter->setText(QString::number(jsonObject["numIter"].toInt()));
            tolIter->setText(QString::number(jsonObject["tolIter"].toDouble()));
        }
      }
     result = true;
  } else {
     return false;
  }

  return result;
}

bool
PLoMInputWidget::copyFiles(QString &fileDir) {
    if (preTrained) {
        qDebug() << inpFileDir2->text();
        qDebug() << fileDir + QDir::separator() + "surrogatePLoM.h5";
        QFile::copy(inpFileDir2->text(), fileDir + QDir::separator() + "surrogatePLoM.h5");
        qDebug() << inpFileDir2->text().replace(".h5",".json");
        qDebug() << fileDir + QDir::separator() + "surrogatePLoM.json";
        QFile::copy(inpFileDir2->text().replace(".h5",".json"), fileDir + QDir::separator() + "surrogatePLoM.json");
    } else {
        QFile::copy(inpFileDir->text(), fileDir + QDir::separator() + "inpFile.in");
        QFile::copy(outFileDir->text(), fileDir + QDir::separator() + "outFile.in");
    }
    if (theConstraintsButton->isChecked()) {
        QFile::copy(constraintsPath->text(), fileDir + QDir::separator() + "plomConstraints.py");
    }
    return true;
}

void
PLoMInputWidget::clear(void)
{

}

int
PLoMInputWidget::getNumberTasks()
{
  return numSamples;
}
