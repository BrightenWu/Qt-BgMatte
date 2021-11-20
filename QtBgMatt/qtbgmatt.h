#pragma once

#include <QtWidgets/QWidget>
#include <QAbstractVideoSurface>
#include <QMediaPlayer> 
#include <QFuture> 
#include "ui_qtbgmatt.h"
#include "bg_matte.h"

class QCamera;

class QRVMWidget :public QWidget
{
public:
	QRVMWidget(QWidget *parent = Q_NULLPTR):QWidget(parent){}
	~QRVMWidget() = default;

	void setImageInfo(int nWidth, int nHeight, QImage::Format eFormat);
	
	std::vector<uint8_t> m_vResCache;

protected:
	void paintEvent(QPaintEvent *event) override;

private:
	QImage::Format m_eImageFormat = QImage::Format_ARGB32;
	int m_nImageWidth = 0;
	int m_nImageHeight = 0;
};

class QVideoSurface : public QAbstractVideoSurface
{
	Q_OBJECT

public:
	QVideoSurface(QObject *parent = Q_NULLPTR) : QAbstractVideoSurface(parent) {}
	~QVideoSurface() = default;

	QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType = QAbstractVideoBuffer::NoHandle) const
	{
		QList<QVideoFrame::PixelFormat> listPixelFormats;

		listPixelFormats << QVideoFrame::Format_ARGB32
			<< QVideoFrame::Format_ARGB32_Premultiplied
			<< QVideoFrame::Format_RGB32
			<< QVideoFrame::Format_RGB24
			<< QVideoFrame::Format_RGB565
			<< QVideoFrame::Format_RGB555
			<< QVideoFrame::Format_ARGB8565_Premultiplied
			<< QVideoFrame::Format_BGRA32
			<< QVideoFrame::Format_BGRA32_Premultiplied
			<< QVideoFrame::Format_BGR32
			<< QVideoFrame::Format_BGR24
			<< QVideoFrame::Format_BGR565
			<< QVideoFrame::Format_BGR555
			<< QVideoFrame::Format_BGRA5658_Premultiplied
			<< QVideoFrame::Format_AYUV444
			<< QVideoFrame::Format_AYUV444_Premultiplied
			<< QVideoFrame::Format_YUV444
			<< QVideoFrame::Format_YUV420P
			<< QVideoFrame::Format_YV12
			<< QVideoFrame::Format_UYVY
			<< QVideoFrame::Format_YUYV
			<< QVideoFrame::Format_NV12
			<< QVideoFrame::Format_NV21
			<< QVideoFrame::Format_IMC1
			<< QVideoFrame::Format_IMC2
			<< QVideoFrame::Format_IMC3
			<< QVideoFrame::Format_IMC4
			<< QVideoFrame::Format_Y8
			<< QVideoFrame::Format_Y16
			<< QVideoFrame::Format_Jpeg
			<< QVideoFrame::Format_CameraRaw
			<< QVideoFrame::Format_AdobeDng;

		// Return the formats you will support
		return listPixelFormats;
	}

	bool present(const QVideoFrame &frame)
	{
		// Handle the frame and do your processing
		if (frame.isValid())
		{
			QVideoFrame cloneFrame(frame);
			emit frameAvailable(cloneFrame);

			return true;
		}

		return false;
	}

signals:
	void frameAvailable(QVideoFrame &frame);
};

class QtBgMatt : public QWidget
{
    Q_OBJECT

public:
    QtBgMatt(QWidget *parent = Q_NULLPTR);
	~QtBgMatt();

private:
	void setConnection();

private:
    Ui::QtBgMattClass ui;
	QString m_strLastDirectory;
	QByteArrayList m_listSpare;
	QByteArrayList m_listBuffer;
	QFuture<void> m_future;

	std::unique_ptr<bgmatt::CMatte> m_pBgMatte;
	std::unique_ptr<bgmatt::CMatte> m_pVideoMatte;
	QVideoSurface *m_pCameraSurface = nullptr;
	QVideoSurface *m_pVideoSurface = nullptr;
	QMediaPlayer *m_pMediaPlayer = nullptr;
	QCamera *m_pCamera = nullptr;
	QRVMWidget *m_pRVMWidget = nullptr;
	bool m_bMatting = false;
	bool m_bExitThread = false;
};
