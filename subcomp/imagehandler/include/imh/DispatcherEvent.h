#pragma once

#include <Image.h>
#include <memory>
#include <Types.h>

namespace neneta
{
namespace imh
{

struct DispatcherEvent
{
    enum Type
    {
        NA,
        STARTOFDB,
        STARTOFBATCH,
        IMAGE,
        ENDOFBATCH,
        ENDOFDB
    };
    virtual Type getEventType() const { return Type::NA; }
};

struct StartOfDbEvent : public DispatcherEvent
{
    Type getEventType() const { return Type::STARTOFDB; }
};

struct StartOfBatchEvent : public DispatcherEvent
{
    Type getEventType() const { return Type::STARTOFBATCH; }
};

struct EndOfBatchEvent : public DispatcherEvent
{
    Type getEventType() const { return Type::ENDOFBATCH; }
};

struct EndOfDbEvent : public DispatcherEvent
{
    Type getEventType() const { return Type::ENDOFDB; }
};

template<typename T>
struct ImageEvent : public DispatcherEvent
{
    typedef ip::Image<T, std::int16_t> Image;
    ImageEvent(std::unique_ptr<Image> img) : m_data(std::move(img)) {}
    Type getEventType() const { return Type::IMAGE; }
    std::unique_ptr<Image> m_data;
};

template struct ImageEvent<cmn::GPUFLOAT>;
template struct ImageEvent<std::uint8_t>;
}
}
