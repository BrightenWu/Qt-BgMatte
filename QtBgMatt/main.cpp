#include "qtbgmatt.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QtBgMatt w;
	w.show();
	return a.exec();
}
