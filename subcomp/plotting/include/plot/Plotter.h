#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <gnuplot-iostream/gnuplot-iostream.h>
#include <tuple>
#include <condition_variable>
#include <Types.h>

namespace neneta
{

namespace conf
{
class ConfigurationReader;
}

namespace plot
{

struct Plotter
{
    Plotter(const conf::ConfigurationReader& confReader);
    ~Plotter();

    void run(Plotter* object);
    void stop();
    void plotNewPoint(cmn::GPUFLOAT y);
    void startNewPlotLine();

    std::atomic<bool> m_run;
    std::condition_variable m_newDataCV;
    std::condition_variable m_plottingEndCV;
    bool m_newData;
    std::string m_cmd;
    gnuplotio::Gnuplot m_gnuplot;
    std::vector<std::vector<std::tuple<int ,cmn::GPUFLOAT>>> m_vectors;
    std::mutex m_mutex;
    std::thread m_plotThread;
};

}
}
