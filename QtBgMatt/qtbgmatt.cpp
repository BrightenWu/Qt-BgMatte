#include "qtbgmatt.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QCameraInfo>
#include <QMetaType> 
#include <QPainter>
#include <QtConcurrent>
#include <QVideoSurfaceFormat>

Q_DECLARE_METATYPE(QCameraInfo)

static constexpr uint8_t FRAME_BUFFER_SIZE = 7;

//////////////////////////////////////////////////////////////////////////

void QRVMWidget::setImageInfo(int nWidth, int nHeight, QImage::Format eFormat)
{
	m_nImageWidth = nWidth;
	m_nImageHeight = nHeight;
	m_eImageFormat = eFormat;
}

void QRVMWidget::paintEvent(QPaintEvent * event)
{
	__super::paintEvent(event);

	//! Resizing maybe cause crash. Try to use QOpenGLWidget to repaint. 
	if (!m_vResCache.empty())
	{
		QPainter painter(this);

		auto img = QImage(reinterpret_cast<uchar*>(m_vResCache.data()), m_nImageWidth, m_nImageHeight, m_eImageFormat).mirrored(false, true);
		painter.drawImage(QRect(0, 0, img.width(), img.height()), img);
	}
}

//////////////////////////////////////////////////////////////////////////

QtBgMatt::QtBgMatt(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);

	m_pRVMWidget = new QRVMWidget(this);
	m_pRVMWidget->setAttribute(Qt::WA_OpaquePaintEvent);
	ui.pGridLayoutRVM->addWidget(m_pRVMWidget, 0, 1, 2, 1);

	m_pBgMatte = bgmatt::CreateMatteObj(bgmatt::ModuleType::MT_BGM);
	m_pVideoMatte = bgmatt::CreateMatteObj(bgmatt::ModuleType::MT_VIDEOM);

	if (!m_pBgMatte->LoadModuleFile("torchscript_mobilenetv2_fp16.pth"))
	{
		QMessageBox::critical(this, "Error", "Cuda or the model file torchscript_mobilenetv2_fp16.pth is not availabel!");
	}

	if (!m_pVideoMatte->LoadModuleFile("rvm_mobilenetv3_fp16.torchscript"))
	{
		QMessageBox::critical(this, "Error", "Cuda or the model file rvm_mobilenetv3_fp16.torchscript is not availabel!");
	}

	m_pCameraSurface = new QVideoSurface(this);
	m_pVideoSurface = new QVideoSurface(this);

	setConnection();
}

QtBgMatt::~QtBgMatt()
{
	m_bExitThread = true;
	m_future.waitForFinished();
}

void QtBgMatt::setConnection()
{
	connect(ui.pButtonTargetBgrImage, &QPushButton::clicked, [this] {
		auto strFilePath = QFileDialog::getOpenFileName(this, tr("Select image"), m_strLastDirectory, tr("image(*.jpg *.jpeg *.png)"));
		if (!strFilePath.isEmpty())
		{
			m_strLastDirectory= QFileInfo(strFilePath).absolutePath();
			ui.pWidgetTargetBgrImage->setPixmap(QPixmap(strFilePath));
			m_pBgMatte->SetTargetBgrImage(QImage(strFilePath));
			m_pVideoMatte->SetTargetBgrImage(QImage(strFilePath));
		}
	});

	connect(ui.pButtonSrcBgrImage, &QPushButton::clicked, [this] {
		auto strFilePath = QFileDialog::getOpenFileName(this, tr("Select image"), m_strLastDirectory, tr("image(*.jpg *.jpeg *.png)"));
		if (!strFilePath.isEmpty())
		{
			m_strLastDirectory = QFileInfo(strFilePath).absolutePath();
			ui.pWidgetSrcBgrImage->setPixmap(QPixmap(strFilePath));
			m_pBgMatte->SetSrcBgrImage(QImage(strFilePath));
		}
	});

	connect(ui.pButtonSrcImage, &QPushButton::clicked, [this] {
		auto strFilePath = QFileDialog::getOpenFileName(this, tr("Select image"), m_strLastDirectory, tr("image(*.jpg *.jpeg *.png)"));
		if (!strFilePath.isEmpty())
		{
			m_strLastDirectory = QFileInfo(strFilePath).absolutePath();
			ui.pWidgetSrcImage->setPixmap(QPixmap(strFilePath));
			auto imgRes = m_pBgMatte->SetImage(QImage(strFilePath));
			ui.pWidgetResultImage->setPixmap(QPixmap::fromImage(imgRes));
		}
	});

	connect(ui.pButtonSearchCameras, &QPushButton::clicked, [this] {
		const auto &listAvailableCameras = QCameraInfo::availableCameras();
		for (const auto &cameraInfo : listAvailableCameras)
		{
			ui.pComboBoxCamera->addItem(cameraInfo.description(), QVariant::fromValue(cameraInfo));
		}
	});

	connect(ui.pButtonOpen, &QPushButton::clicked, [this] {
		auto data = ui.pComboBoxCamera->currentData();
		if (data.isValid())
		{
			m_pCamera = new QCamera(data.value<QCameraInfo>(), this);
			m_pCamera->setViewfinder(m_pCameraSurface);
			m_pCamera->start();
		}
	});

	connect(ui.pButtonClose, &QPushButton::clicked, [this] {
		if (m_pCamera)
		{
			m_pCamera->stop();
		}
	});

	connect(ui.pButtonTargetBgrVideo, &QPushButton::clicked, [this] {
		auto strFilePath = QFileDialog::getOpenFileName(this, tr("Select video"), m_strLastDirectory, tr("video(*.mp4 *.mov *.flv)"));
		if (!strFilePath.isEmpty())
		{
			m_strLastDirectory = QFileInfo(strFilePath).absolutePath();

			if (!m_pMediaPlayer)
			{
				m_pMediaPlayer = new QMediaPlayer(this);
				m_pMediaPlayer->setVideoOutput(m_pVideoSurface);
			}
			else
			{
				m_pMediaPlayer->stop();
			}

			m_pMediaPlayer->setMedia(QUrl(strFilePath));
			m_pMediaPlayer->play();
		}
	});

	connect(ui.pButtonMatte, &QPushButton::clicked, [this] (bool checked){
		m_bMatting = checked;

		if (m_listSpare.isEmpty())
		{
			for (uint8_t i = 0; i < FRAME_BUFFER_SIZE; i++)
			{
				m_listSpare.push_back(QByteArray());
			}

			m_future = QtConcurrent::run([this]() {
				while (!m_bExitThread)
				{
					if (m_bMatting && 
						QCamera::ActiveStatus == m_pCamera->status() && 
						m_pCameraSurface->isActive() && 
						!m_listBuffer.isEmpty())
					{
						const auto &formatVideoSurface = m_pCameraSurface->surfaceFormat();
						auto byteArrayBuffer = m_listBuffer.front();
						
						auto imgRes = m_pVideoMatte->SetImage(
							QImage(reinterpret_cast<uchar*>(byteArrayBuffer.data()),
										  formatVideoSurface.frameWidth(), 
										  formatVideoSurface.frameHeight(),
										  QVideoFrame::imageFormatFromPixelFormat(formatVideoSurface.pixelFormat())));

						auto nSize = imgRes.bytesPerLine()*imgRes.height();
						if (m_pRVMWidget->m_vResCache.size() != nSize)
						{
							m_pRVMWidget->m_vResCache.resize(nSize);
						}

						memcpy(m_pRVMWidget->m_vResCache.data(), imgRes.bits(), nSize);

						m_pRVMWidget->update();
					}
				}
			});
		}
	});

	connect(m_pCameraSurface, &QVideoSurface::frameAvailable, [&](QVideoFrame &frame) {
		if (frame.map(QAbstractVideoBuffer::ReadOnly))
		{
			if (!m_bMatting)
			{
				m_pRVMWidget->setImageInfo(frame.width(), frame.height(), QVideoFrame::imageFormatFromPixelFormat(frame.pixelFormat()));

				auto nSize = frame.bytesPerLine()*frame.height();
				if (m_pRVMWidget->m_vResCache.size() != nSize)
				{
					m_pRVMWidget->m_vResCache.resize(nSize);
				}

				memcpy(m_pRVMWidget->m_vResCache.data(), frame.bits(), nSize);

				m_pRVMWidget->update();
			}
			else
			{
				auto byteArraySpare = m_listSpare.front();
				m_listSpare.pop_front();

				if (!byteArraySpare.isEmpty())
				{
					byteArraySpare.clear();
				}

				byteArraySpare = std::move(QByteArray(reinterpret_cast<char *>(frame.bits()), frame.bytesPerLine()*frame.height()));
				m_listBuffer.push_back(byteArraySpare);
				if (m_listBuffer.size() + 2 > FRAME_BUFFER_SIZE)
				{
					auto byteArray = m_listBuffer.front();
					m_listBuffer.pop_front();
					m_listSpare.push_back(byteArray);
				}
			}

			frame.unmap();
		}
	});

	connect(m_pVideoSurface, &QVideoSurface::frameAvailable, [&](QVideoFrame &frame) {
		if (frame.map(QAbstractVideoBuffer::ReadOnly) && m_bMatting)
		{
			auto recvImage = QImage(frame.bits(), frame.width(), frame.height(), QVideoFrame::imageFormatFromPixelFormat(frame.pixelFormat())).mirrored(false, true);
			m_pVideoMatte->SetTargetBgrImage(recvImage);

			frame.unmap();
		}
	});
}
