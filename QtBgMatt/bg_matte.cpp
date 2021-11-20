#include <torch/script.h>
#include "bg_matte.h"
#include <torch/csrc/api/include/torch/cuda.h>
#include <QFile>

namespace bgmatt
{
	class CMattePrivate
	{
	public:
		CMattePrivate() = default;
		virtual ~CMattePrivate() = default;

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
		torch::Tensor m_tensorTargetBgr;
		QByteArray m_arrayResCache;

		bgmatt::MatteResolution m_eMatteResolution = bgmatt::MatteResolution::MR_HD;
		torch::Device m_sDevice = torch::Device("cuda");
		c10::ScalarType m_nPrecision = torch::kFloat16;
	};

	class CBgMattePrivate :public CMattePrivate
	{
	public:
		CBgMattePrivate() = default;
		~CBgMattePrivate() = default;

		torch::Tensor m_tensorSrcBgr;
	};

	class CRVMMattePrivate :public CMattePrivate
	{
	public:
		CRVMMattePrivate() = default;
		~CRVMMattePrivate() = default;

		c10::optional<torch::Tensor> m_tensorRec0;
		c10::optional<torch::Tensor> m_tensorRec1;
		c10::optional<torch::Tensor> m_tensorRec2;
		c10::optional<torch::Tensor> m_tensorRec3;
		float m_fDownsampleRatio = 0.4;
	};

	//////////////////////////////////////////////////////////////////////////

	CMatte::CMatte()
	{
		d_ptr = std::make_shared<CMattePrivate>();
		d_ptr->m_tensorTargetBgr = torch::tensor({ 120.f / 255, 255.f / 255, 155.f / 255 }).toType(d_ptr->m_nPrecision).to(d_ptr->m_sDevice).view({ 1, 3, 1, 1 });
	}

	MatteResolution CMatte::GetMatteResolution() const
	{
		return d_ptr->m_eMatteResolution;
	}

	void CMatte::SetTargetBgrImage(const QImage & imgTargetBgr)
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

		return;
	}

	QImage CMatte::SetImage(const QString &strSrcAbsolutePath, const QString &strBgrAbsolutePath)
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

	CMatte::CMatte(std::shared_ptr<CMattePrivate> d) :d_ptr(d)
	{
		d_ptr->m_tensorTargetBgr = torch::tensor({ 120.f / 255, 255.f / 255, 155.f / 255 }).toType(d_ptr->m_nPrecision).to(d_ptr->m_sDevice).view({ 1, 3, 1, 1 });
	}

	//////////////////////////////////////////////////////////////////////////

	CBgMatte::CBgMatte():CMatte(std::make_shared<CBgMattePrivate>())
	{

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

	void CBgMatte::SetMatteResolution(MatteResolution eR)
	{
		switch (eR)
		{

		case MatteResolution::MR_HD:
		{
			if (d_ptr->m_sModel.hasattr("refine_mode"))
			{
				d_ptr->m_sModel.setattr("backbone_scale", 0.25);
				d_ptr->m_sModel.setattr("refine_sample_pixels", 80000);
			}
		}
			break;

		case MatteResolution::MR_4K:
		{
			if (d_ptr->m_sModel.hasattr("refine_mode"))
			{
				d_ptr->m_sModel.setattr("backbone_scale", 0.125);
				d_ptr->m_sModel.setattr("refine_sample_pixels", 320000);
			}
		}
			break;

		case MatteResolution::MR_SD:
		default:
			Q_ASSERT_X(0, __FUNCTION__, "Type error!");
			break;
		}

		d_ptr->m_eMatteResolution = eR;
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

		auto pBgmatte = std::dynamic_pointer_cast<CBgMattePrivate>(d_ptr);
		pBgmatte->m_tensorSrcBgr = tensorBg.to(d_ptr->m_nPrecision).div(255);
		pBgmatte->m_tensorSrcBgr.unsqueeze_(0);

		return true;
	}

	QImage CBgMatte::SetImage(const QImage & imgSrc)
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

		//auto start = std::chrono::high_resolution_clock::now();

		//! Inference
		torch::NoGradGuard no_grad;
		auto pBgmatte = std::dynamic_pointer_cast<CBgMattePrivate>(d_ptr);
		auto outputs = d_ptr->m_sModel.forward({ tensorSrc, pBgmatte->m_tensorSrcBgr }).toTuple()->elements();

		//auto time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());

		auto pha = outputs[0].toTensor();
		auto fgr = outputs[1].toTensor();

		auto res_tensor = pha * fgr + (1 - pha) * d_ptr->m_tensorTargetBgr;
		res_tensor = res_tensor.mul(255).to(torch::kUInt8).cpu().permute({ 0,2,3,1 });
		res_tensor.squeeze_(0);
		res_tensor = res_tensor.contiguous();

		d_ptr->m_arrayResCache = QByteArray(static_cast<char *>(res_tensor.data_ptr()), res_tensor.size(1)*res_tensor.size(0)*res_tensor.size(2));
		QImage imgRes(reinterpret_cast<uchar *>(d_ptr->m_arrayResCache.data()), res_tensor.size(1), res_tensor.size(0), QImage::Format_RGB888);

		return imgRes.convertToFormat(formatSrc);
	}

	//////////////////////////////////////////////////////////////////////////

	CRVMMatte::CRVMMatte() :CMatte(std::make_shared<CRVMMattePrivate>())
	{

	}

	bool CRVMMatte::LoadModuleFile(const QString & strModuleAbsolutePath)
	{
		if (!d_ptr->IsCudaAvailable() || !QFile::exists(strModuleAbsolutePath))
		{
			return false;
		}

		d_ptr->m_sModel = torch::jit::load(strModuleAbsolutePath.toStdString());

		//! Optionally, freeze the model. This will trigger graph optimization, such as BatchNorm fusion etc. Frozen models are faster.
		//torch::jit::freeze(d_ptr->m_sModel);
		d_ptr->m_sModel.to(d_ptr->m_sDevice);
		SetMatteResolution(d_ptr->m_eMatteResolution);

		return true;
	}

	void CRVMMatte::SetMatteResolution(MatteResolution eR)
	{
		auto pBgmatte = std::dynamic_pointer_cast<CRVMMattePrivate>(d_ptr);

		switch (eR)
		{
		case MatteResolution::MR_SD:
			pBgmatte->m_fDownsampleRatio = 0.6;
			break;

		case MatteResolution::MR_HD:
			pBgmatte->m_fDownsampleRatio = 0.4;
		break;

		case MatteResolution::MR_4K:
			pBgmatte->m_fDownsampleRatio = 0.2;
		break;

		default:
			Q_ASSERT_X(0, __FUNCTION__, "Type error!");
			break;
		}

		d_ptr->m_eMatteResolution = eR;
	}

	QImage CRVMMatte::SetImage(const QImage & imgSrc)
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

		//! Inference
		torch::NoGradGuard no_grad;

		auto pBgmatte = std::dynamic_pointer_cast<CRVMMattePrivate>(d_ptr);
		auto outputs = d_ptr->m_sModel.forward({
			tensorSrc,
			pBgmatte->m_tensorRec0,
			pBgmatte->m_tensorRec1,
			pBgmatte->m_tensorRec2,
			pBgmatte->m_tensorRec3,
			pBgmatte->m_fDownsampleRatio }).toList();

		const auto &fgr = outputs.get(0).toTensor();
		const auto &pha = outputs.get(1).toTensor();
		pBgmatte->m_tensorRec0 = outputs.get(2).toTensor();
		pBgmatte->m_tensorRec1 = outputs.get(3).toTensor();
		pBgmatte->m_tensorRec2 = outputs.get(4).toTensor();
		pBgmatte->m_tensorRec3 = outputs.get(5).toTensor();

		auto res_tensor = pha * fgr + (1 - pha) * d_ptr->m_tensorTargetBgr;

		res_tensor = res_tensor.mul(255).permute({ 0,2,3,1 })[0].to(torch::kU8).contiguous().cpu();

		d_ptr->m_arrayResCache = QByteArray(static_cast<char *>(res_tensor.data_ptr()), res_tensor.size(1)*res_tensor.size(0)*res_tensor.size(2));
		QImage imgRes(reinterpret_cast<uchar *>(d_ptr->m_arrayResCache.data()), res_tensor.size(1), res_tensor.size(0), QImage::Format_RGB888);

		return imgRes.convertToFormat(formatSrc);
	}

	//////////////////////////////////////////////////////////////////////////

	std::unique_ptr<CMatte> CreateMatteObj(ModuleType eType)
	{
		std::unique_ptr<CMatte> p;
		switch (eType)
		{
		case bgmatt::ModuleType::MT_BGM:
			p = std::make_unique<CBgMatte>();
			break;

		case bgmatt::ModuleType::MT_VIDEOM:
			p = std::make_unique<CRVMMatte>();
			break;

		default:
			break;
		}

		return p;
	}
}
