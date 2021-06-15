#include "qtbgmatt.h"
//#include <torch/script.h>
//#include <torch/csrc/api/include/torch/cuda.h>
//#include <QImage>
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
#if 0
	std::cout << "cuda是否可用：" << torch::cuda::is_available() << std::endl;
	std::cout << "cudnn是否可用：" << torch::cuda::cudnn_is_available() << std::endl;
	
	//! Load model
	auto device = torch::Device("cuda");
	auto precision = torch::kFloat32;

	auto model = torch::jit::load(R"(E:\JetBrains\PyCharm 2018.3.5\works\bgmatt\model\TorchScript\torchscript_mobilenetv2_fp32.pth)");

	model.setattr("backbone_scale", 0.25);
	model.setattr("refine_mode", "sampling");
	model.setattr("refine_sample_pixels", 80000);
	model.to(device);

	//! Load image
	QImage imgSrc(R"(E:\Microsoft Visual Studio\2017\Enterprise\works\QtBgMatt\x64\Release\input_img\src\src1.png)");
	QImage imgBg(R"(E:\Microsoft Visual Studio\2017\Enterprise\works\QtBgMatt\x64\Release\input_img\bg\bg1.png)");

	//! Convert BGRA to RGB
	imgSrc = imgSrc.convertToFormat(QImage::Format_RGB888);
	imgBg = imgBg.convertToFormat(QImage::Format_RGB888);

	auto tensorSrc = torch::from_blob(imgSrc.bits(), { imgSrc.height(),imgSrc.width(),3 }, torch::kByte);
	tensorSrc = tensorSrc.to(device);
	tensorSrc = tensorSrc.permute({ 2,0,1 }).contiguous();
	auto tmpSrc = tensorSrc.to(precision).div(255);
	tmpSrc.unsqueeze_(0);
	tmpSrc = tmpSrc.to(precision);

	auto tensorBg = torch::from_blob(imgBg.bits(), { imgBg.height(),imgBg.width(),3 }, torch::kByte);
	tensorBg = tensorBg.to(device);
	tensorBg = tensorBg.permute({ 2,0,1 }).contiguous();
	auto tmpBg = tensorBg.to(precision).div(255);
	tmpBg.unsqueeze_(0);
	tmpBg = tmpBg.to(precision);

	//std::ofstream fout(L"tmpSrc.txt", std::ios_base::out /*| std::ios_base::app*/);
	//if (fout.is_open())
	//{
	//	fout << tmpSrc << std::endl;
	//	fout.close();
	//}

	//! Inference
	auto start = std::chrono::high_resolution_clock::now();

	//torch::NoGradGuard no_grad;
	auto outputs = model.forward({ tmpSrc, tmpBg }).toTuple()->elements();

	std::cout << "time(ms):" << static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count()) << std::endl;

	auto pha = outputs[0].toTensor();
	auto fgr = outputs[1].toTensor();

	auto pha_size = pha.sizes();
	std::cout << "pha_size size " << pha_size << std::endl;
	auto fgr_size = fgr.sizes();
	std::cout << "fgr_size size " << fgr_size << std::endl;

	auto tgt_bgr = torch::tensor({ 120.f / 255, 255.f / 255, 155.f / 255 }).toType(precision).to(device).view({ 1, 3, 1, 1 });
	std::cout << "tgt_bgr size " << tgt_bgr.sizes() << std::endl;

	auto res_tensor = pha * fgr + (1 - pha) * tgt_bgr;
	res_tensor = res_tensor.mul(255).to(torch::kUInt8).cpu().permute({ 0,2,3,1 });
	res_tensor.squeeze_(0);
	res_tensor = res_tensor.contiguous();

	std::cout << "res_tensor size " << res_tensor.sizes() << std::endl;

	QImage imgRes(static_cast<uchar *>(res_tensor.data_ptr()), res_tensor.size(1), res_tensor.size(0), QImage::Format_RGB888);
	std::cout << "Save: " << imgRes.save("Res.png") << std::endl;
#endif

	QApplication a(argc, argv);
	QtBgMatt w;
	w.show();
	return a.exec();
}
