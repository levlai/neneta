#include <gtest/gtest.h>
#include <Utils.h>
#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <OpenCLExecutionPlan.h>


using namespace neneta;
using namespace neneta::net;


extern conf::ConfigurationReader envReader;

TEST(BackPropagationTest, basic_bp_test)
{
    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);


    std::vector<cmn::GPUFLOAT> reInput = {1, 2};
    std::vector<cmn::GPUFLOAT> imInput = {3, 4};

    //two neurons
    std::vector<cmn::GPUFLOAT> reWeights = {0, 0};
    std::vector<cmn::GPUFLOAT> imWeights = {0, 0};
    std::vector<cmn::GPUFLOAT> reActDer = {1, 2};
    std::vector<cmn::GPUFLOAT> imActDer = {1, 1};

    //right two neurons, 10+j9 and 10+j8 are the deltas    
    const unsigned int numOfNeuronsLeft = 2;
    std::vector<cmn::GPUFLOAT> rightReWeights = {0, 1, 2, 3, 10, 10, 109, 108};
    std::vector<cmn::GPUFLOAT> rightImWeights = {1, 2, 1, 2, 9, 8, 199, 198};

    if(oclProgram.compile(oclContext))
    {

        gpu::OpenCLExecutionPlan uut("test", envReader, oclProgram);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rightReWeights.size()*sizeof(cmn::GPUFLOAT), rightReWeights.data());
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rightReWeights.size()*sizeof(cmn::GPUFLOAT), rightImWeights.data());
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, rightReWeights.size()*sizeof(cmn::GPUFLOAT));
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, rightReWeights.size()*sizeof(cmn::GPUFLOAT));


        LayerWeights weightsToUpdate;
        weightsToUpdate.emplace_back(cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reWeights.size()*sizeof(cmn::GPUFLOAT), reWeights.data()),
                                     cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reWeights.size()*sizeof(cmn::GPUFLOAT), imWeights.data()), reWeights.size());
        weightsToUpdate.emplace_back(cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reWeights.size()*sizeof(cmn::GPUFLOAT), reWeights.data()),
                                     cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reWeights.size()*sizeof(cmn::GPUFLOAT), imWeights.data()), reWeights.size());


        LayerInput layerInput(cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data()),
                              cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reInput.size()*sizeof(cmn::GPUFLOAT), imInput.data()), reInput.size());

        LayerDeltas LayerDeltas(cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reActDer.size()*sizeof(cmn::GPUFLOAT), reActDer.data()),
                                                  cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reActDer.size()*sizeof(cmn::GPUFLOAT), imActDer.data()), reActDer.size());

        LayerBiases layerBiases(cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, numOfNeuronsLeft*sizeof(cmn::GPUFLOAT)),
                                cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, numOfNeuronsLeft*sizeof(cmn::GPUFLOAT)), numOfNeuronsLeft);


        //net::updateWeights(uut, input, weightsToUpdate, LayerDeltas, layerBiases, layerInput,
         //                 weightsToUpdate.size(), oclContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

        uut.runBckPropagation(oclContext);
        uut.printBckProfilingInfo();

        int j = 0;
        /* result should be
         * neuron 1
            31578 + j18326
            40544 + j31972
           neuron 2
            77494 + j87808
            90930 + j138430

        neuron 0 delta = -6823 + j-4483
        neuron 1 delta = -13843 + j-9093
        */
        for(const auto& neuronWeights: weightsToUpdate)
        {
            std::cout << "neuron " << ++j << std::endl;
            cmn::GPUFLOAT* rePart = new cmn::GPUFLOAT[neuronWeights.m_size];
            cmn::GPUFLOAT* imPart = new cmn::GPUFLOAT[neuronWeights.m_size];
            uut.readFromBuffer(oclContext.getCommandQueue(), neuronWeights.m_re, sizeof(cmn::GPUFLOAT)*neuronWeights.m_size, rePart);
            uut.readFromBuffer(oclContext.getCommandQueue(), neuronWeights.m_im, sizeof(cmn::GPUFLOAT)*neuronWeights.m_size, imPart);
            for(unsigned int i = 0; i < neuronWeights.m_size; ++i)
            {
                std::cout << "\t" << rePart[i] << " + j" << imPart[i]<< std::endl;
            }
            delete [] rePart;
            delete [] imPart;
        }

        //print out deltas
        cmn::GPUFLOAT* rePart = new cmn::GPUFLOAT[numOfNeuronsLeft];
        cmn::GPUFLOAT* imPart = new cmn::GPUFLOAT[numOfNeuronsLeft];
        uut.readFromBuffer(oclContext.getCommandQueue(), LayerDeltas.m_re, sizeof(cmn::GPUFLOAT)*numOfNeuronsLeft, rePart);
        uut.readFromBuffer(oclContext.getCommandQueue(), LayerDeltas.m_im, sizeof(cmn::GPUFLOAT)*numOfNeuronsLeft, imPart);

        for(unsigned int i  = 0; i < numOfNeuronsLeft; ++i)
        {
            std::cout << "neuron " << i << " delta = " << rePart[i] << " + j" << imPart[i] << std::endl;
        }
        delete [] rePart;
        delete [] imPart;

        //check reshm
        rePart = new cmn::GPUFLOAT[rightReWeights.size()];
        imPart = new cmn::GPUFLOAT[rightReWeights.size()];
        uut.readFromBuffer(oclContext.getCommandQueue(), input.m_reShMem, sizeof(cmn::GPUFLOAT)*rightReWeights.size(), rePart);
        uut.readFromBuffer(oclContext.getCommandQueue(), input.m_imShMem, sizeof(cmn::GPUFLOAT)*rightReWeights.size(), imPart);

        for(unsigned int i  = 0; i < rightReWeights.size(); ++i)
        {
            std::cout << "weight " << i << std::endl;
            std::cout << "\t" << rePart[i] << " + j" << imPart[i]<< std::endl;
        }
        delete [] rePart;
        delete [] imPart;


    }
}
