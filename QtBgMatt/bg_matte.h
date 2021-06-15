#pragma once
#include <QImage>

namespace bgmatt
{
	class CBgMattePrivate;

	class CBgMatte
	{
	public:
		CBgMatte();
		virtual ~CBgMatte() = default;

		enum MatteResolution
		{
			MR_HD,
			MR_4K
		};

		void SetMatteResolution(MatteResolution eR);
		MatteResolution GetMatteResolution() const;

		bool LoadModuleFile(const QString &strModuleAbsolutePath);

		QImage SetImage(const QString &strSrcAbsolutePath, const QString &strBgrAbsolutePath);

		bool SetSrcBgrImage(const QImage &imgBgr);
		void SetTargetBgrImage(const QImage &imgTargetBgr);
		QImage SetSrcImage(const QImage &imgSrc);

	private:
		std::unique_ptr<CBgMattePrivate> d_ptr;
	};
}