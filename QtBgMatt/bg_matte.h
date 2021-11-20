/************************************************************************
Programmer: BrightenWu
Date: 2021.11.16
Issue&P.S.:
BackgroundMattingV2:
1. backbone_scale (float, default: 0.25): The downsampling scale that the backbone should operate on. e.g,
the backbone will operate on 480x270 resolution for a 1920x1080 input with backbone_scale=0.25.
2. refine_sample_pixels (int, default: 80,000). The fixed amount of pixels to refine. Used in sampling mode.
3. We recommend backbone_scale=0.25, refine_sample_pixels=80000 for HD and backbone_scale=0.125, refine_sample_pixels=320000 for 4K.
4. https://github.com/PeterL1n/BackgroundMattingV2/blob/master/doc/model_usage.md

 RobustVideoMatting:
1. Downsample Ratio
	Resolution 	Portrait 	Full-Body
	<= 512x512 	1 				1
	1280x720 	0.375 			0.6
	1920x1080 	0.25 			0.4
	3840x2160 	0.125 			0.2
2. https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md
************************************************************************/

#pragma once
#include <QImage>

namespace bgmatt
{
	class CMattePrivate;

	enum class MatteResolution
	{
		MR_SD,  //!< 1280x720
		MR_HD,  //!< 1920x1080
		MR_4K  //!< 3840x2160
	};

	enum class ModuleType
	{
		MT_BGM,  //!< BackgroundMattingV2
		MT_VIDEOM  //!< RobustVideoMatting
	};

	class CMatte
	{
	public:
		CMatte();
		virtual ~CMatte() = default;

		virtual bool LoadModuleFile(const QString &strModuleAbsolutePath) = 0;

		virtual void SetMatteResolution(MatteResolution eR) = 0;
		MatteResolution GetMatteResolution() const;

		void SetTargetBgrImage(const QImage &imgTargetBgr);

		//! only BackgroundMattingV2
		virtual bool SetSrcBgrImage(const QImage &imgBgr) { return false; }

		//! Get matted image
		virtual QImage SetImage(const QImage &imgSrc) = 0;

		[[deprecated]] QImage SetImage(const QString &strSrcAbsolutePath, const QString &strBgrAbsolutePath);

	protected:
		CMatte(std::shared_ptr<CMattePrivate> d);

	protected:
		std::shared_ptr<CMattePrivate> d_ptr;
	};

	class CBgMatte :public CMatte
	{
	public:
		CBgMatte();
		~CBgMatte() = default;

		bool LoadModuleFile(const QString &strModuleAbsolutePath) override;

		void SetMatteResolution(MatteResolution eR) override;

		bool SetSrcBgrImage(const QImage &imgBgr) override;
		QImage SetImage(const QImage &imgSrc) override;
	};

	class CRVMMatte :public CMatte
	{
	public:
		CRVMMatte();
		~CRVMMatte() = default;

		bool LoadModuleFile(const QString &strModuleAbsolutePath) override;

		void SetMatteResolution(MatteResolution eR) override;

		QImage SetImage(const QImage &imgSrc) override;
	};

	std::unique_ptr<CMatte> CreateMatteObj(ModuleType eType);
}