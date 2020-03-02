#include <Plotter.h>
#include <ConfigurationReader.h>

using namespace neneta::plot;

Plotter::Plotter(const conf::ConfigurationReader& confReader)
    : m_run(true)
    , m_newData(false)
    , m_cmd(confReader.getStringParameter("configuration.plotting.gnuplotcmd"))
    , m_gnuplot(m_cmd)
    , m_plotThread(&Plotter::run, this, this)
{                
    startNewPlotLine();
}

Plotter::~Plotter()
{
    stop();
}

void Plotter::run(Plotter* object)
{    
    while(object->m_run)
    {
        std::unique_lock<std::mutex> guard(object->m_mutex);
        object->m_newDataCV.wait(guard, [object]{ return object->m_newData || !object->m_run;});
        for(size_t ln = 0; ln < object->m_vectors.size(); ++ln)
        {
            if(ln == 0)
            {
                object->m_gnuplot << "plot ";
            }
            object->m_gnuplot << "'-' with lines title 'Epoch " << ln << "'";
            if(ln == object->m_vectors.size()-1)
            {
                object->m_gnuplot << "\n";
            }
            else
            {
                object->m_gnuplot << ", ";
            }
        }

        for(size_t ln = 0; ln < object->m_vectors.size(); ++ln)
        {
            object->m_gnuplot.send1d(object->m_vectors[ln]);
        }
        object->m_gnuplot.flush();
        m_newData = false;
        guard.unlock();
        object->m_plottingEndCV.notify_one();

    }    
}

void Plotter::stop()
{    
    {
        std::unique_lock<std::mutex> guard(m_mutex);
        m_plottingEndCV.wait(guard, [this](){return !m_newData;});
        m_run = false;
    }
    m_newDataCV.notify_one();
    if(m_plotThread.joinable())
    {
        m_plotThread.join();
    }
}

void Plotter::plotNewPoint(cmn::GPUFLOAT y)
{
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_vectors.back().emplace_back(std::make_tuple(m_vectors.back().size(),y));
        m_newData = true;
    }
    m_newDataCV.notify_one();
}


void Plotter::startNewPlotLine()
{
    m_vectors.emplace_back();
}
