#pragma once

#include <QtWidgets/QWidget>
#include "ui_qtbgmatt.h"

class QtBgMatt : public QWidget
{
    Q_OBJECT

public:
    QtBgMatt(QWidget *parent = Q_NULLPTR);

private:
    Ui::QtBgMattClass ui;
	QImage imgRes;
};
