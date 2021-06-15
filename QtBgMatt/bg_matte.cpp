#include <torch/script.h>
#include "bg_matte.h"
#include <torch/csrc/api/include/torch/cuda.h>
#include <QFile>

namespace bgmatt
{
	class CBgMattePrivate
	{
	public:
		CBgMattePrivate() = default;
		~CBgMattePrivate() = default;

		bool IsCudaAvailable() const
		{
			auto bCuda = torch::cuda::is_available();
			auto bCudnn = torch::cuda::cudnn_is_available();
			if (!bCuda || !bCudnn)
			{
				return false;
			}

			return true;
		}

		torch::jit::Module m_sModel;
		torch::Device m_sDevice = torch::Device("cuda");
		torch::Tensor m_tensorSrcBgr;
		torch::Tensor m_tensorTargetBgr;
		c10::ScalarType m_nPrecision = torch::kFloat32;
		CBgMatte::MatteResolution m_eMatteResolution = CBgMatte::MatteResolution::MR_HD;
		QByteArray m_arrayResCache;
	};

	//////////////////////////////////////////////////////////////////////////

	CBgMatte::CBgMatte()
	{
		d_ptr = std::move(std::unique_ptr<CBgMattePrivate>(new CBgMattePrivate));
	}

	void CBgMatte::SetMatteResolution(MatteResolution eR)
	{
		if (!d_ptr->m_sModel.hasattr("refine_mode"))
		{
			Q_ASSERT_X(0, __FUNCTION__, "Module error!");
			return;
		}

		switch (d_ptr->m_eMatteResolution)
		{
		case MR_HD:
			d_ptr->m_sModel.setattr("backbone_scale", 0.25);
			d_ptr->m_sModel.setattr("refine_sample_pixels", 80000);
			break;
		case MR_4K:
			d_ptr->m_sModel.setattr("backbone_scale", 0.125);
			d_ptr->m_sModel.setattr("refine_sample_pixels", 320000);
			break;
		default:
			Q_ASSERT_X(0, __FUNCTION__, "Type error!");
			break;
		}
	}

	CBgMatte::MatteResolution CBgMatte::GetMatteResolution() const
	{
		return d_ptr->m_eMatteResolution;
	}

	bool CBgMatte::LoadModuleFile(const QString &strModuleAbsolutePath)
	{
		if (!d_ptr->IsCudaAvailable() || !QFile::exists(strModuleAbsolutePath))
		{
			return false;
		}

		d_ptr->m_sModel = torch::jit::load(strModuleAbsolutePath.toStdString());
		d_ptr->m_sModel.setattr("refine_mode", "sampling");
		d_ptr->m_sModel.to(d_ptr->m_sDevice);

		SetMatteResolution(d_ptr->m_eMatteResolution);
		return true;
	}

	QImage CBgMatte::SetImage(const QString &strSrcAbsolutePath, const QString &strBgrAbsolutePath)
	{
		if (!d_ptr->IsCudaAvailable())
		{
			return QImage();
		}

		//! Load image
		QImage imgSrc(strSrcAbsolutePath);
		QImage imgBg(strBgrAbsolutePath);

		if (imgSrc.isNull() || imgBg.isNull())
		{
			return QImage();
		}

		auto formatBg = imgBg.format();

		//! Convert formatBg to RGB
		imgSrc = imgSrc.convertToFormat(QImage::Format_RGB888);
		imgBg = imgBg.convertToFormat(QImage::Format_RGB888);

		auto tensorSrc = torch::from_blob(imgSrc.bits(), { imgSrc.height(),imgSrc.width(),3 }, torch::kByte);
		tensorSrc = tensorSrc.to(d_ptr->m_sDevice);
		tensorSrc = tensorSrc.permute({ 2,0,1 }).contiguous();
		auto tmpSrc = tensorSrc.to(d_ptr->m_nPrecision).div(255);
		tmpSrc.unsqueeze_(0);
		tmpSrc = tmpSrc.to(d_ptr->m_nPrecision);

		auto tensorBg = torch::from_blob(imgBg.bits(), { imgBg.height(),imgBg.width(),3 }, torch::kByte);
		tensorBg = tensorBg.to(d_ptr->m_sDevice);
		tensorBg = tensorBg.permute({ 2,0,1 }).contiguous();
		auto tmpBg = tensorBg.to(d_ptr->m_nPrecision).div(255);
		tmpBg.unsqueeze_(0);
		tmpBg = tmpBg.to(d_ptr->m_nPrecision);

		//! Inference
		//torch::NoGradGuard no_grad;
		auto outputs = d_ptr->m_sModel.forward({ tmpSrc, tmpBg }).toTuple()->elements();

		auto pha = outputs[0].toTensor();
		auto fgr = outputs[1].toTensor();
		auto tgt_bgr = torch::tensor({ 120.f / 255, 255.f / 255, 155.f / 255 }).toType(d_ptr->m_nPrecision).to(d_ptr->m_sDevice).view({ 1, 3, 1, 1 });

		auto res_tensor = pha * fgr + (1 - pha) * tgt_bgr;
		res_tensor = res_tensor.mul(255).to(torch::kUInt8).cpu().permute({ 0,2,3,1 });
		res_tensor.squeeze_(0);
		res_tensor = res_tensor.contiguous();

		d_ptr->m_arrayResCache = QByteArray(static_cast<char *>(res_tensor.data_ptr()), res_tensor.size(1)*res_tensor.size(0)*res_tensor.size(2));
		QImage imgRes(reinterpret_cast<uchar *>(d_ptr->m_arrayResCache.data()), res_tensor.size(1), res_tensor.size(0), QImage::Format_RGB888);
		return imgRes.convertToFormat(formatBg);
	}

	bool CBgMatte::SetSrcBgrImage(const QImage & imgBgr)
	{
		if (!d_ptr->IsCudaAvailable())
		{
			return false;
		}

		if (imgBgr.isNull())
		{
			return false;
		}

		//! Load image
		QImage imgBg(imgBgr);

		//! Convert to RGB
		imgBg = imgBg.convertToFormat(QImage::Format_RGB888);

		auto tensorBg = torch::from_blob(imgBg.bits(), { imgBg.height(),imgBg.width(),3 }, torch::kByte);
		tensorBg = tensorBg.to(d_ptr->m_sDevice);
		tensorBg = tensorBg.permute({ 2,0,1 }).contiguous();
		d_ptr->m_tensorSrcBgr = tensorBg.to(d_ptr->m_nPrecision).div(255);
		d_ptr->m_tensorSrcBgr.unsqueeze_(0);
		d_ptr->m_tensorSrcBgr = d_ptr->m_tensorSrcBgr.to(d_ptr->m_nPrecision);

		return true;
	}

	void CBgMatte::SetTargetBgrImage(const QImage & imgTargetBgr)
	{
		if (imgTargetBgr.isNull())
		{
			d_ptr->m_tensorTargetBgr = torch::tensor({ 120.f / 255, 255.f / 255, 155.f / 255 }).toType(d_ptr->m_nPrecision).to(d_ptr->m_sDevice).view({ 1, 3, 1, 1 });
			return;
		}

		//! Load image
		QImage imgBg(imgTargetBgr);

		//! Convert to RGB
		imgBg = imgBg.convertToFormat(QImage::Format_RGB888);

		auto tensorBg = torch::from_blob(imgBg.bits(), { imgBg.height(),imgBg.width(),3 }, torch::kByte);
		tensorBg = tensorBg.to(d_ptr->m_sDevice);
		tensorBg = tensorBg.permute({ 2,0,1 }).contiguous();
		d_ptr->m_tensorTargetBgr = tensorBg.to(d_ptr->m_nPrecision).div(255);
		d_ptr->m_tensorTargetBgr.unsqueeze_(0);
		d_ptr->m_tensorTargetBgr = d_ptr->m_tensorTargetBgr.to(d_ptr->m_nPrecision);

		return;
	}

	QImage CBgMatte::SetSrcImage(const QImage & imgSrc)
	{
		if (!d_ptr->IsCudaAvailable())
		{
			return QImage();
		}

		if (imgSrc.isNull())
		{
			return QImage();
		}

		//! Load image
		QImage imgSrcCopy(imgSrc);

		auto formatSrc = imgSrcCopy.format();

		//! Convert formatBg to RGB
		imgSrcCopy = imgSrcCopy.convertToFormat(QImage::Format_RGB888);
		
		auto tensorSrc = torch::from_blob(imgSrcCopy.bits(), { imgSrcCopy.height(),imgSrcCopy.width(),3 }, torch::kByte);
		tensorSrc = tensorSrc.to(d_ptr->m_sDevice);
		tensorSrc = tensorSrc.permute({ 2,0,1 }).contiguous();
		tensorSrc = tensorSrc.to(d_ptr->m_nPrecision).div(255);
		tensorSrc.unsqueeze_(0);
		tensorSrc = tensorSrc.to(d_ptr->m_nPrecision);

		//auto start = std::chrono::high_resolution_clock::now();

		//! Inference
		torch::NoGradGuard no_grad;
		auto outputs = d_ptr->m_sModel.forward({ tensorSrc, d_ptr->m_tensorSrcBgr }).toTuple()->elements();

		//auto time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());

		auto pha = outputs[0].toTensor();
		auto fgr = outputs[1].toTensor();
		//auto tgt_bgr = torch::tensor({ 120.f / 255, 255.f / 255, 155.f / 255 }).toType(d_ptr->m_nPrecision).to(d_ptr->m_sDevice).view({ 1, 3, 1, 1 });

		auto res_tensor = pha * fgr + (1 - pha) * d_ptr->m_tensorTargetBgr;
		res_tensor = res_tensor.mul(255).to(torch::kUInt8).cpu().permute({ 0,2,3,1 });
		res_tensor.squeeze_(0);
		res_tensor = res_tensor.contiguous();

		d_ptr->m_arrayResCache = QByteArray(static_cast<char *>(res_tensor.data_ptr()), res_tensor.size(1)*res_tensor.size(0)*res_tensor.size(2));
		QImage imgRes(reinterpret_cast<uchar *>(d_ptr->m_arrayResCache.data()), res_tensor.size(1), res_tensor.size(0), QImage::Format_RGB888);

		return imgRes.convertToFormat(formatSrc);
	}
}
