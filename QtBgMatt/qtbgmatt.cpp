#include "qtbgmatt.h"
#include "bg_matte.h"
#include <QMessageBox>

QtBgMatt::QtBgMatt(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);

	std::unique_ptr<bgmatt::CBgMatte> pMatte(new bgmatt::CBgMatte);
	if (!pMatte->LoadModuleFile("torchscript_mobilenetv2_fp32.pth"))
	{
		QMessageBox::critical(this, "Error", "Cuda or the model file is not availabel!");
	}
	else
	{
		imgRes = pMatte->SetImage(R"(input_img\src\src1.png)", R"(input_img\bg\bg1.png)");
		//imgRes = imgRes.convertToFormat(QImage::Format_ARGB32);
		//imgRes.save("Res.png");

		auto palette = this->palette();
		palette.setBrush(QPalette::Window, QBrush(QPixmap::fromImage(imgRes)));
		this->setPalette(palette);
	}
}
