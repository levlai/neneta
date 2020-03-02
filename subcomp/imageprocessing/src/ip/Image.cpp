#include <Image.h>
#include <boost/log/trivial.hpp>
#include <sstream>

using namespace neneta;
using namespace neneta::ip;
using namespace cimg_library;

template<typename T, typename Label>
Image<T, Label>::Image(const std::string& imgPath, bool gs) : m_label(0)
{
    try
    {
        m_image.load(imgPath.c_str());
        if(gs)
        {
            m_image = m_image.channel(0);
        }
    }
    catch(const cimg_library::CImgIOException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << ex.what();
    }
}

template<typename T, typename Label>
Image<T, Label>::Image(const T* data, const int size)
    : m_image(data, size, size), m_label(0)
{
}

template<typename T, typename Label>
Image<T, Label>::Image(const T* data, const int width, const int height, Label label)
    : m_image(data, width, height)
    , m_label(label)
{
}

template<typename T, typename Label>
Image<T, Label>::Image(const Image<T, Label>& img) : m_image(img.m_image), m_label(img.m_label)
{
}

template<typename T, typename Label>
Image<T, Label>::Image(const cimg_library::CImg<T>& img, Label label) : m_image(img), m_label(label)
{
}

template<typename T, typename Label>
template<typename U>
Image<T, Label>& Image<T, Label>::operator=(const Image<U, Label>& img)
{
    m_image = img.m_image;
    m_label = img.m_label;
    return *this;
}

template<typename T, typename Label>
Image<T, Label>::Image(Image&& img) : m_image(std::move(img.m_image)), m_label(img.m_label)
{
}

template<typename T, typename Label>
Image<T, Label>& Image<T, Label>::operator=(Image&& img)
{
    m_image = std::move(img.m_image);
    m_label = img.m_label;
    return *this;
}

template<typename T, typename Label>
Image<T, Label>::~Image()
{
}

template<typename T, typename Label>
bool Image<T, Label>::resizeWithCrop(int sizexy)
{
    bool rv = false;
    if(m_image.spectrum() == 3 || m_image.depth() == 1)
    {
        if(m_image.width() != sizexy || m_image.height() != sizexy)
        {
            int diff = sizexy - std::min(m_image.width(), m_image.height());
            if(diff != 0)
            {
                m_image.resize(m_image.width() + diff, m_image.height() + diff, 1, 3, 6);
            }

            if(m_image.width() < m_image.height())
            {
                int diff = (m_image.height() - m_image.width())/2;
                m_image.crop(0, diff, m_image.width() - 1, diff + m_image.width() - 1);
            }
            else if(m_image.width() > m_image.height())
            {   int diff = (m_image.width() - m_image.height())/2;
                m_image.crop(diff, 0, diff + m_image.height() - 1, m_image.height() - 1);
            }
        }
        rv = true;
    }
    return rv;
}

template<typename T, typename Label>
bool Image<T, Label>::resize(int sizexy)
{
    bool rv = false;
    if(m_image.spectrum() == 3 && m_image.depth() == 1)
    {
        if(m_image.width() != sizexy || m_image.height() != sizexy)
        {
            m_image.resize(sizexy, sizexy, 1, 3, 6);
        }
        rv = true;
    }
    else if(m_image.spectrum() == 1 && m_image.depth() == 1)
    {
        if(m_image.width() != sizexy || m_image.height() != sizexy)
        {
            m_image.resize(sizexy, sizexy, 1, 1, 6);
        }
        rv = true;
    }
    else
    {
        assert("unsupported image type");
    }
    return rv;
}

template<typename T, typename Label>
void Image<T, Label>::normalize(const int min, const int max)
{
    m_image.normalize(min,max);
}

template<typename T, typename Label>
void Image<T, Label>::show() const
{
    CImgDisplay display;
    display.assign(m_image);
    display.show();
}

template<typename T, typename Label>
void Image<T, Label>::showAndWait() const
{    
    CImgDisplay display(m_image, std::to_string(m_label).c_str());
    while (!display.is_closed())
    {
      display.wait();
   }
    display.close();    
}

template<typename T, typename Label>
size_t Image<T, Label>::width() const
{
    return m_image.width();
}

template<typename T, typename Label>
size_t Image<T, Label>::height() const
{
    return m_image.height();
}

template<typename T, typename Label>
size_t Image<T, Label>::channels() const
{
    return m_image.spectrum();
}

template<typename T, typename Label>
const cimg_library::CImg<T>& Image<T, Label>::getCImg() const
{
    return m_image;
}

template<typename T, typename Label>
T* Image<T, Label>::data(unsigned int channel)
{
    return m_image.data(0,0,0,channel);
}

template<typename T, typename Label>
void Image<T, Label>::zeroPad(const int padding)
{   
    m_image.resize(m_image.width() + padding, m_image.height() + padding, m_image.depth(), m_image.spectrum(), 0);
}

template<typename T, typename Label>
void Image<T, Label>::printImage(unsigned int z, unsigned int channel)
{
    for(int i = 0; i < m_image.height(); ++i)
    {
        std::stringstream ss;
        for(int j = 0; j < m_image.width(); ++j)
        {

            ss << static_cast<T>(m_image(j,i,z,channel)) << ",\t";
        }
        BOOST_LOG_TRIVIAL(debug) << ss.str();
    }
}
/*
template<typename T, typename Label>
Image<cmn::GPUFLOAT, Label> Image<T, Label>::convertToComplex(const std::uint16_t maxInt) const
{
    cimg_library::CImg<cmn::GPUFLOAT> complexImage(m_image.width(), m_image.height(), 1, 2, 0);
    cimg_library::CImg<cmn::GPUFLOAT> re = complexImage.get_shared_channel(0);
    cimg_library::CImg<cmn::GPUFLOAT> im = complexImage.get_shared_channel(1);
    re = (1/static_cast<cmn::GPUFLOAT>(maxInt))*m_image;
    im = 0*m_image;
    return Image<cmn::GPUFLOAT, Label>(complexImage, m_label);
}
*/
template<typename T, typename Label>
Image<cmn::GPUFLOAT, Label> Image<T, Label>::convertToComplex(const std::uint16_t maxInt) const
{
    std::uint16_t max = m_image.max();
    cimg_library::CImg<cmn::GPUFLOAT> complexImage(m_image.width(), m_image.height(), 1, 2, 0);
    cimg_library::CImg<cmn::GPUFLOAT> re = complexImage.get_shared_channel(0);
    cimg_library::CImg<cmn::GPUFLOAT> im = complexImage.get_shared_channel(1);
    re = (2*PI/static_cast<cmn::GPUFLOAT>(max))*m_image;
    im = (2*PI/static_cast<cmn::GPUFLOAT>(max))*m_image;
    re.cos();
    im.sin();
    return Image<cmn::GPUFLOAT, Label>(complexImage, m_label);
}

template<typename T, typename Label>
Image<cmn::GPUFLOAT, Label> Image<T, Label>::convertToRealFloat(const std::uint16_t maxInt) const
{
    cimg_library::CImg<cmn::GPUFLOAT> complexImage(m_image.width(), m_image.height(), 1, 2, 0);
    cimg_library::CImg<cmn::GPUFLOAT> re = complexImage.get_shared_channel(0);
    re = (1.0f/static_cast<cmn::GPUFLOAT>(maxInt))*m_image;
    return Image<cmn::GPUFLOAT, Label>(complexImage, m_label);
}

template class Image<cmn::GPUFLOAT, std::int16_t>;
template class Image<std::uint8_t, std::int16_t>;


